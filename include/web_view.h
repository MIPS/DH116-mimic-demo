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
// It listens on all network interfaces without a password: anyone who reaches the port can watch and send keys
// (in snapshot mode: press Capture, which moves the hand).
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <atomic>
#include <cerrno>
#include <cstdint>
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
    explicit WebView(int port, bool snapshot = false) : snapshot_(snapshot)
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

    // True once per press of Capture, for the main loop to start that capture.
    bool take_capture()
    {
        int expected = REQUESTED;
        return state_.compare_exchange_strong(expected, BUSY);
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

        if (get && route == "/stream" && streams.size() < MAX_STREAMS && send_all(fd, STREAM_HEADER))
        {
            streams.push_back(fd);
            return;
        }
        if (get && route == "/") send_all(fd, reply("200 OK", "text/html", snapshot_ ? SNAPSHOT_PAGE : PAGE));
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
            if (request_capture()) send_all(fd, reply("200 OK", "text/plain", "ok\n"));
            else send_all(fd, reply("409 Conflict", "text/plain", "busy\n"));
        }
        else if (get && route == "/stream") send_all(fd, reply("503 Service Unavailable", "text/plain", "too many viewers\n"));
        else send_all(fd, reply("404 Not Found", "text/plain", "not found\n"));
        ::close(fd);
    }

    // Locks the Capture button and shows "Inferencing" at once; false if a capture is already running.
    bool request_capture()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        int expected = READY;
        if (!state_.compare_exchange_strong(expected, REQUESTED)) return false;
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
    // when the server says so.
    static constexpr const char* SNAPSHOT_PAGE = R"html(<!doctype html>
<html><head><meta charset="utf-8"><title>DH116 Mimic - snapshot</title>
<style>
body{margin:0;background:#111;color:#ddd;font:16px sans-serif;text-align:center}
.panes{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:8px}
@media(max-width:800px){.panes{grid-template-columns:1fr}}
.pane{position:relative;background:#000;aspect-ratio:4/3}
.pane img{width:100%;height:100%;object-fit:contain;display:block}
.pane b{position:absolute;left:8px;top:6px;padding:2px 8px;border-radius:4px;background:#000a;font-weight:normal;font-size:14px}
#shot{display:none}
#none{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#888}
button{font-size:24px;margin:8px;padding:12px 40px}
#msg{min-height:1.6em;padding-bottom:12px;font-size:20px}
.flash{color:#ffb74d;animation:blink .8s ease-in-out infinite}
@keyframes blink{50%{opacity:.2}}
</style></head>
<body>
<div class="panes">
<div class="pane"><b>Live</b><img src="/stream" alt="live camera"></div>
<div class="pane"><b>Last capture</b><img id="shot" alt="last capture"><div id="none">No capture yet</div></div>
</div>
<button id="go" disabled>Capture</button>
<div id="msg">Connecting...</div>
<script>
const goBtn = document.getElementById('go'), msgEl = document.getElementById('msg');
const shotImg = document.getElementById('shot'), noneEl = document.getElementById('none');
let seen = 0, polling = false, epoch = 0;
async function poll() {
  if (polling) return;
  polling = true;
  const mine = epoch;
  try {
    const s = await (await fetch('/state')).json();
    if (mine !== epoch) return; // the button was pressed meanwhile: this answer may be older than the press
    goBtn.disabled = !s.ready;
    msgEl.textContent = s.text;
    msgEl.className = s.flash ? 'flash' : '';
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
  msgEl.textContent = 'Inferencing...';
  msgEl.className = 'flash';
  epoch++;
  try { await fetch('/capture', {method: 'POST'}); } catch (e) {}
  epoch++;
  goBtn.blur();
  poll();
};
setInterval(poll, 250);
poll();
</script></body></html>
)html";

    enum State { READY, REQUESTED, BUSY }; // snapshot mode: Capture pressed -> main loop took it -> finish()

    const bool snapshot_;
    int listen_fd_ = -1;
    int wake_[2] = {-1, -1}; // pipe that wakes the server thread for a new image or to stop
    std::atomic<bool> running_{true};
    std::atomic<int> key_{-1};
    std::atomic<int> state_{READY};
    std::mutex mutex_;
    cv::Mat image_;      // newest image; guarded by mutex_
    uint64_t count_ = 0; // images published; guarded by mutex_
    std::string status_ = "Ready. Press Capture."; // snapshot mode, guarded by mutex_
    bool flash_ = false;
    std::string shot_;       // JPEG of the last capture's result
    uint64_t shot_id_ = 0;   // results published; 0 = none yet
    std::thread thread_;
};
