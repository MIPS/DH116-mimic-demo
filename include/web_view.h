// web_view.h - shows the hand_cam preview in a web browser (hand_cam --web PORT).
//
// One server thread answers three requests:
//   GET /         a page with the video and the O/F/R buttons (the keys also work on the page)
//   GET /stream   the video as MJPEG: one HTTP response in which each new frame replaces the previous one
//   GET /key/X    calibration key X (o, f or r) for the main loop, as if pressed in the preview window
// WebView(port, true) is the snapshot mode of hand_cam --snapshot. / is then a page with the live video, a
// Capture button and the result of the last capture, there are no keys, and these requests are added:
//   POST /capture the Capture button: asks the main loop for one capture; 409 while the last one is not finished
//   GET /state    JSON for the page: whether Capture can be pressed, a status text, the number of the last result
//   GET /shot.jpg the result of the last capture
// WebView(port, true, width, height) is the snapshot mode with the camera of the browser (hand_cam --camera browser):
// the page shows that camera, asked for at width x height, instead of /stream, and each press of Capture sends the
// picture it took as a JPEG in the body of POST /capture. The server decodes it for the main loop (take_capture).
// It listens on all network interfaces without a password: anyone who reaches the port can watch and send keys
// (in snapshot mode: press Capture, which moves the hand; with the browser camera: choose the picture it moves for).
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

class WebView
{
public:
    // camera_width > 0 (snapshot mode only): the pictures come from the browser's camera, not from this machine.
    explicit WebView(int port, bool snapshot = false, int camera_width = 0, int camera_height = 0)
        : snapshot_(snapshot), camera_width_(camera_width), camera_height_(camera_height)
    {
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<uint16_t>(port));
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
        const int on = 1;
        listen_fd_ = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
        if (listen_fd_ < 0 || setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on)) < 0 ||
            bind(listen_fd_, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) < 0 ||
            listen(listen_fd_, 8) < 0 || pipe2(wake_, O_NONBLOCK | O_CLOEXEC) < 0)
        {
            const std::string error = std::strerror(errno);
            ::close(listen_fd_);
            throw std::runtime_error("--web port " + std::to_string(port) + ": " + error);
        }
        thread_ = std::thread(&WebView::serve, this);
    }
    ~WebView()
    {
        running_ = false;
        wake();
        thread_.join();
        ::close(listen_fd_);
        ::close(wake_[0]);
        ::close(wake_[1]);
    }
    WebView(const WebView&) = delete;
    WebView& operator=(const WebView&) = delete;

    // Hands over the newest preview image; the server thread encodes it and sends it to every open /stream.
    void publish(const cv::Mat& image)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            image_ = image.clone(); // new pixels, so an image still being encoded is not overwritten
            ++count_;
        }
        wake();
    }

    // The key last sent from a browser ('o', 'f' or 'r'), or -1.
    int take_key() { return key_.exchange(-1); }

    // Snapshot mode. One capture runs from a press of Capture to finish(); the button is locked in between,
    // so a press (or a second browser) while one is running changes nothing.

    // True once per press of Capture, for the main loop to start that capture. With the browser camera, `picture`
    // (BGR, as the browser took it) gets the picture that came with the press.
    bool take_capture(cv::Mat* picture = nullptr)
    {
        int expected = REQUESTED;
        if (!state_.compare_exchange_strong(expected, BUSY)) return false;
        if (picture)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            *picture = upload_;
            upload_.release();
        }
        return true;
    }

    // The text under the Capture button; flash makes it blink (used for "Inferencing").
    void set_status(const std::string& text, bool flash = false)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        status_ = text;
        flash_ = flash;
    }

    // Replaces the picture in the "last capture" panel; the JPEG is encoded here, in the caller's thread.
    void set_shot(const cv::Mat& image)
    {
        std::vector<uchar> jpeg;
        cv::imencode(".jpg", image, jpeg, {cv::IMWRITE_JPEG_QUALITY, JPEG_QUALITY});
        std::lock_guard<std::mutex> lock(mutex_);
        shot_.assign(jpeg.begin(), jpeg.end());
        ++shot_id_;
    }

    // Ends the capture: shows the final text and unlocks the Capture button.
    void finish(const std::string& text)
    {
        set_status(text);
        state_ = READY;
    }

private:
    static constexpr size_t MAX_STREAMS = 4; // browsers watching at once
    static constexpr int JPEG_QUALITY = 75;
    // Browser camera: limits for the request that carries one picture.
    static constexpr size_t MAX_UPLOAD = 16u << 20;                    // bytes of the JPEG
    static constexpr long MAX_PIXELS = 16'000'000;                     // width x height the JPEG claims
    static constexpr size_t MAX_HEADER = 16 * 1024;                    // bytes of request headers
    static constexpr std::chrono::seconds UPLOAD_TIMEOUT{5};           // for the whole request

    bool browser_camera() const { return snapshot_ && camera_width_ > 0; }

    void wake()
    {
        const char byte = 0;
        [[maybe_unused]] const ssize_t n = ::write(wake_[1], &byte, 1); // fails only if a wake-up is pending
    }

    // Waits for new connections and new images, and sends each new image once to every open /stream.
    void serve()
    {
        std::vector<int> streams;
        uint64_t sent = 0; // number of the last image sent
        while (running_)
        {
            pollfd fds[2] = {{listen_fd_, POLLIN, 0}, {wake_[0], POLLIN, 0}};
            if (poll(fds, 2, -1) < 0 && errno != EINTR) break;
            char drain[64];
            while (::read(wake_[0], drain, sizeof(drain)) > 0) {}
            if (fds[0].revents & POLLIN)
            {
                const int fd = ::accept4(listen_fd_, nullptr, nullptr, SOCK_CLOEXEC);
                if (fd >= 0) answer(fd, streams);
            }

            cv::Mat image;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (count_ == sent || streams.empty()) continue;
                image = image_;
                sent = count_;
            }
            std::vector<uchar> jpeg;
            cv::imencode(".jpg", image, jpeg, {cv::IMWRITE_JPEG_QUALITY, JPEG_QUALITY});
            std::string part = "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + std::to_string(jpeg.size()) +
                               "\r\n\r\n";
            part.append(jpeg.begin(), jpeg.end());
            part += "\r\n";
            for (auto it = streams.begin(); it != streams.end();)
            {
                if (send_all(*it, part)) ++it;
                else
                {
                    ::close(*it); // browser closed, or stalled for longer than the send timeout
                    it = streams.erase(it);
                }
            }
        }
        for (const int fd : streams) ::close(fd);
    }

    // Reads one request. A /stream connection stays open in `streams`; the others are answered and closed.
    void answer(int fd, std::vector<int>& streams)
    {
        const timeval timeout{0, 500000}; // the longest a stalled browser can hold up the server thread
        const int buffer = 128 * 1024;    // a few frames: a slow browser misses frames instead of lagging behind
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
        setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buffer, sizeof(buffer));

        char request[1024];
        const ssize_t n = ::recv(fd, request, sizeof(request), 0);
        const std::string line(request, n > 0 ? static_cast<size_t>(n) : 0);
        std::string method, path; // from "GET /path?query HTTP/1.1"
        const size_t space = line.find(' ');
        if (space != std::string::npos)
        {
            method = line.substr(0, space);
            path = line.substr(space + 1, line.find(' ', space + 1) - space - 1);
        }
        const std::string route = path.substr(0, path.find('?'));
        const bool get = method == "GET";

        if (get && route == "/stream" && !browser_camera() && streams.size() < MAX_STREAMS && send_all(fd, STREAM_HEADER))
        {
            streams.push_back(fd);
            return;
        }
        if (get && route == "/") send_all(fd, reply("200 OK", "text/html", snapshot_ ? snapshot_page() : PAGE));
        else if (get && !snapshot_ && route.size() == 6 && route.rfind("/key/", 0) == 0 &&
                 std::string("ofr").find(route[5]) != std::string::npos)
        {
            key_ = route[5];
            send_all(fd, reply("200 OK", "text/plain", "ok\n"));
        }
        else if (get && snapshot_ && route == "/state") send_all(fd, reply("200 OK", "application/json", state_json()));
        else if (get && snapshot_ && route == "/shot.jpg")
        {
            std::string jpeg;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                jpeg = shot_;
            }
            if (jpeg.empty()) send_all(fd, reply("404 Not Found", "text/plain", "no capture yet\n"));
            else send_all(fd, reply("200 OK", "image/jpeg", jpeg));
        }
        else if (method == "POST" && snapshot_ && route == "/capture")
        {
            cv::Mat picture; // empty unless the browser's camera took it
            std::string error;
            if (browser_camera() && !read_picture(fd, line, picture, error)) send_all(fd, error);
            else if (request_capture(picture)) send_all(fd, reply("200 OK", "text/plain", "ok\n"));
            else send_all(fd, reply("409 Conflict", "text/plain", "busy\n"));
        }
        else if (get && route == "/stream" && !browser_camera())
            send_all(fd, reply("503 Service Unavailable", "text/plain", "too many viewers\n"));
        else send_all(fd, reply("404 Not Found", "text/plain", "not found\n"));
        ::close(fd);
    }

    // Browser camera: reads the picture of a POST /capture into `picture`. `data` is what the first recv returned;
    // the rest of the headers and then the Content-Length bytes of the body follow. False, with the whole reply in
    // `error`, unless that is a complete JPEG of a sensible size that decodes.
    static bool read_picture(int fd, std::string data, cv::Mat& picture, std::string& error)
    {
        const auto fail = [&](const char* status, const char* text)
        {
            error = reply(status, "text/plain", std::string(text) + "\n");
            return false;
        };
        const timeval timeout{2, 0}; // the longest pause inside the upload; UPLOAD_TIMEOUT limits the sum
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        const auto deadline = std::chrono::steady_clock::now() + UPLOAD_TIMEOUT;
        char chunk[16384];
        const auto receive = [&](std::string& into, size_t most)
        {
            if (std::chrono::steady_clock::now() > deadline) return false;
            const ssize_t n = ::recv(fd, chunk, std::min(sizeof(chunk), most), 0);
            if (n <= 0) return false;
            into.append(chunk, static_cast<size_t>(n));
            return true;
        };

        size_t end; // of the headers
        while ((end = data.find("\r\n\r\n")) == std::string::npos)
        {
            if (data.size() > MAX_HEADER)
                return fail("431 Request Header Fields Too Large", "The request headers are too long.");
            if (!receive(data, sizeof(chunk))) return fail("408 Request Timeout", "The request did not arrive in time.");
        }
        std::string head(data, 0, end);
        for (char& c : head) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        const size_t at = head.find("\r\ncontent-length:");
        const unsigned long long announced =
            at == std::string::npos ? 0 : std::strtoull(head.c_str() + at + std::strlen("\r\ncontent-length:"), nullptr, 10);
        if (announced == 0) return fail("400 Bad Request", "There is no picture in the request.");
        if (announced > MAX_UPLOAD) return fail("413 Payload Too Large", "The picture is too big.");

        const size_t length = static_cast<size_t>(announced);
        std::string body(data, end + 4, length); // the part that came with the headers
        body.reserve(length);
        while (body.size() < length)
            if (!receive(body, length - body.size()))
                return fail("408 Request Timeout", "The picture did not arrive in time.");

        long width = 0, height = 0;
        if (!jpeg_size(body, width, height)) return fail("415 Unsupported Media Type", "The picture must be a JPEG.");
        if (width * height > MAX_PIXELS) return fail("413 Payload Too Large", "The picture has too many pixels.");
        try // a tiny file can claim a huge picture, so the size was checked first
        {
            picture = cv::imdecode(cv::Mat(1, static_cast<int>(body.size()), CV_8UC1, body.data()), cv::IMREAD_COLOR);
        }
        catch (const cv::Exception&)
        {
            picture.release();
        }
        if (picture.empty()) return fail("400 Bad Request", "The picture could not be decoded.");
        return true;
    }

    // Width and height from the frame header of a JPEG, without decoding it; false if `data` is not a JPEG.
    static bool jpeg_size(const std::string& data, long& width, long& height)
    {
        const auto byte = [&](size_t i) { return static_cast<unsigned char>(data[i]); };
        if (data.size() < 4 || byte(0) != 0xFF || byte(1) != 0xD8) return false; // no start-of-image marker
        for (size_t i = 2; i + 4 <= data.size();)
        {
            if (byte(i) != 0xFF) return false;
            const int marker = byte(i + 1);
            if (marker == 0xFF) // padding before a marker
            {
                ++i;
                continue;
            }
            if (marker == 0x01 || (marker >= 0xD0 && marker <= 0xD8)) // markers that have no length
            {
                i += 2;
                continue;
            }
            const size_t length = (static_cast<size_t>(byte(i + 2)) << 8) | byte(i + 3);
            const bool frame = marker >= 0xC0 && marker <= 0xCF && marker != 0xC4 && marker != 0xC8 && marker != 0xCC;
            if (frame)
            {
                if (i + 9 > data.size()) return false;
                height = (byte(i + 5) << 8) | byte(i + 6);
                width = (byte(i + 7) << 8) | byte(i + 8);
                return true;
            }
            if (marker == 0xDA || length < 2) return false; // the picture data starts without a frame header
            i += 2 + length;
        }
        return false;
    }

    // Locks the Capture button and shows "Inferencing" at once; false if a capture is already running.
    // `upload` is the picture that came with the press (browser camera), for take_capture().
    bool request_capture(const cv::Mat& upload)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        int expected = READY;
        if (!state_.compare_exchange_strong(expected, REQUESTED)) return false;
        upload_ = upload;
        status_ = "Inferencing...";
        flash_ = true;
        return true;
    }

    std::string state_json()
    {
        const bool ready = state_ == READY;
        std::lock_guard<std::mutex> lock(mutex_);
        std::string text;
        for (const unsigned char c : status_) // JSON string: escape quotes and backslashes, no control characters
        {
            if (c == '"' || c == '\\') text += '\\';
            text += c < 0x20 ? ' ' : static_cast<char>(c);
        }
        return std::string("{\"ready\":") + (ready ? "true" : "false") + ",\"flash\":" + (flash_ ? "true" : "false") +
               ",\"text\":\"" + text + "\",\"shot\":" + std::to_string(shot_id_) + "}\n";
    }

    static std::string reply(const std::string& status, const std::string& type, const std::string& body)
    {
        return "HTTP/1.1 " + status + "\r\nContent-Type: " + type + "\r\nContent-Length: " +
               std::to_string(body.size()) + "\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n" + body;
    }

    static bool send_all(int fd, const std::string& data)
    {
        for (size_t done = 0; done < data.size();)
        {
            const ssize_t n = ::send(fd, data.data() + done, data.size() - done, MSG_NOSIGNAL);
            if (n <= 0) return false;
            done += static_cast<size_t>(n);
        }
        return true;
    }

    static constexpr const char* STREAM_HEADER = "HTTP/1.1 200 OK\r\n"
                                                 "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
                                                 "Cache-Control: no-store\r\nConnection: close\r\n\r\n";
    static constexpr const char* PAGE = R"html(<!doctype html>
<html><head><meta charset="utf-8"><title>DH116 Mimic</title>
<style>body{margin:0;background:#111;color:#ddd;font:16px sans-serif;text-align:center}
img{max-width:100%;max-height:88vh}button{font-size:18px;margin:8px}</style></head>
<body><img src="/stream" alt="camera"><br>
<button onclick="key('o')">O: open hand</button><button onclick="key('f')">F: fist</button><button onclick="key('r')">R: reset</button>
<script>
function key(k) { fetch('/key/' + k); }
document.onkeydown = e => { const k = e.key.toLowerCase(); if (k === 'o' || k === 'f' || k === 'r') key(k); };
</script></body></html>
)html";

    // Snapshot mode: the live video and the last result side by side, one Capture button (no keyboard shortcut,
    // so a stray key cannot move the hand). The page asks /state four times a second and unlocks the button only
    // when the server says so. With the browser camera (snapshot_page() fills in CAM) the live pane is the
    // browser's own camera, mirrored like the stream, and Capture sends one picture of it to the server.
    static constexpr const char* SNAPSHOT_PAGE = R"html(<!doctype html>
<html><head><meta charset="utf-8"><title>DH116 Mimic - snapshot</title>
<style>
body{margin:0;background:#111;color:#ddd;font:16px sans-serif;text-align:center}
.panes{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:8px}
@media(max-width:800px){.panes{grid-template-columns:1fr}}
.pane{position:relative;background:#000;aspect-ratio:4/3}
.pane img,.pane video{width:100%;height:100%;object-fit:contain;display:block}
.pane video{transform:scaleX(-1)}
.pane b{position:absolute;right:8px;top:6px;padding:2px 8px;border-radius:4px;background:#000a;font-weight:normal;font-size:14px;z-index:1}
#shot{display:none}
#none{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#888}
button{font-size:24px;margin:8px;padding:12px 40px}
#msg{min-height:1.6em;padding:0 12px 12px;font-size:20px}
.flash{color:#ffb74d;animation:blink .8s ease-in-out infinite}
@keyframes blink{50%{opacity:.2}}
</style></head>
<body>
<div class="panes">
<div class="pane"><b>Live</b><img id="live" alt="live camera"><video id="cam" autoplay muted playsinline></video></div>
<div class="pane"><b>Last capture</b><img id="shot" alt="last capture"><div id="none">No capture yet</div></div>
</div>
<button id="go" disabled>Capture</button>
<div id="msg">Connecting...</div>
<script>
const CAM = /*CAM*/{browser: false};
const goBtn = document.getElementById('go'), msgEl = document.getElementById('msg');
const shotImg = document.getElementById('shot'), noneEl = document.getElementById('none');
const liveImg = document.getElementById('live'), video = document.getElementById('cam');
let seen = 0, polling = false, epoch = 0;
let camError = ''; // browser camera: why it cannot be used
let problem = '';  // browser camera: why the last picture was not accepted; shown until the next press
const camReady = () => !CAM.browser || (!camError && video.videoWidth > 0 && video.readyState >= 2);

async function startCamera() {
  if (!window.isSecureContext || !navigator.mediaDevices) {
    const p = location.port || '80';
    camError = 'A browser allows its camera only on https:// or http://localhost. On this computer run: ssh -L ' + p +
      ':localhost:' + p + ' <user>@' + location.hostname + ' and open http://localhost:' + p + '/';
    return;
  }
  try {
    video.srcObject = await navigator.mediaDevices.getUserMedia(
      {video: {width: {ideal: CAM.width}, height: {ideal: CAM.height}}, audio: false});
  } catch (e) {
    camError = 'The camera is not available (' + e.name + (e.message ? ': ' + e.message : '') +
      '). Allow it for this page and reload.';
  }
}
// One picture of the camera as JPEG: the raw frame, not the mirrored view on the page.
function grab() {
  return new Promise((resolve, reject) => {
    const w = video.videoWidth, h = video.videoHeight;
    if (!camReady()) return reject(new Error('the camera is not ready'));
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    canvas.getContext('2d').drawImage(video, 0, 0, w, h);
    canvas.toBlob(b => b ? resolve(b) : reject(new Error('the picture could not be encoded')), 'image/jpeg', 0.92);
  });
}
// What goes under the button: [text, flashing].
function describe(s) {
  if (!camReady()) return [camError || 'Waiting for the camera. Allow it in the browser if it asks.', false];
  if (problem && s.ready) return [problem, false];
  return [s.text, s.flash];
}
async function poll() {
  if (polling) return;
  polling = true;
  const mine = epoch;
  try {
    const s = await (await fetch('/state')).json();
    if (mine !== epoch) return; // the button was pressed meanwhile: this answer may be older than the press
    const [text, flash] = describe(s);
    goBtn.disabled = !s.ready || !camReady();
    msgEl.textContent = text;
    msgEl.className = flash ? 'flash' : '';
    if (s.shot !== seen) {
      seen = s.shot;
      shotImg.src = '/shot.jpg?n=' + seen;
      shotImg.style.display = 'block';
      noneEl.style.display = 'none';
    }
  } catch (e) {
    goBtn.disabled = true;
    msgEl.textContent = 'No connection to hand_cam';
    msgEl.className = '';
  } finally {
    polling = false;
  }
}
goBtn.onclick = async () => {
  goBtn.disabled = true;
  problem = '';
  msgEl.textContent = 'Inferencing...';
  msgEl.className = 'flash';
  epoch++;
  try {
    const request = {method: 'POST'};
    if (CAM.browser) {
      request.headers = {'Content-Type': 'image/jpeg'};
      request.body = await grab();
    }
    const r = await fetch('/capture', request);
    if (!r.ok && r.status !== 409) problem = (await r.text()).trim() || 'The picture was refused (' + r.status + ').';
  } catch (e) {
    if (CAM.browser) problem = 'Capture failed: ' + e.message + '.';
  }
  epoch++;
  goBtn.blur();
  poll();
};
if (CAM.browser) {
  liveImg.remove();
  startCamera();
} else {
  video.remove();
  liveImg.src = '/stream';
}
setInterval(poll, 250);
poll();
</script></body></html>
)html";

    // The page for the snapshot mode; the server fills in whether the camera is the browser's, and at what size.
    std::string snapshot_page() const
    {
        std::string page = SNAPSHOT_PAGE;
        const std::string off = "/*CAM*/{browser: false}";
        const size_t at = page.find(off);
        if (browser_camera() && at != std::string::npos)
            page.replace(at, off.size(),
                         "{browser: true, width: " + std::to_string(camera_width_) + ", height: " + std::to_string(camera_height_) + "}");
        return page;
    }

    enum State { READY, REQUESTED, BUSY }; // snapshot mode: Capture pressed -> main loop took it -> finish()

    const bool snapshot_;
    const int camera_width_, camera_height_; // browser camera: the size asked of it; 0 = the pictures come from this machine
    int listen_fd_ = -1;
    int wake_[2] = {-1, -1}; // pipe that wakes the server thread for a new image or to stop
    std::atomic<bool> running_{true};
    std::atomic<int> key_{-1};
    std::atomic<int> state_{READY};
    std::mutex mutex_;
    cv::Mat image_;      // newest image; guarded by mutex_
    uint64_t count_ = 0; // images published; guarded by mutex_
    cv::Mat upload_;     // browser camera: the picture of the last press, until take_capture() takes it; guarded by mutex_
    std::string status_ = "Ready. Press Capture."; // snapshot mode, guarded by mutex_
    bool flash_ = false;
    std::string shot_;       // JPEG of the last capture's result
    uint64_t shot_id_ = 0;   // results published; 0 = none yet
    std::thread thread_;
};
