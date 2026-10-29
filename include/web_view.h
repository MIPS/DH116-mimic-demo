// web_view.h - shows the hand_cam preview in a web browser (hand_cam --web PORT).
//
// One server thread answers three requests:
//   GET /         a page with the video and the O/F/R buttons (the keys also work on the page)
//   GET /stream   the video as MJPEG: one HTTP response in which each new frame replaces the previous one
//   GET /key/X    calibration key X (o, f or r) for the main loop, as if pressed in the preview window
// It listens on all network interfaces without a password: anyone who reaches the port can watch and send keys.
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
    explicit WebView(int port)
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
        std::string path; // from "GET /path HTTP/1.1"
        if (line.rfind("GET ", 0) == 0) path = line.substr(4, line.find(' ', 4) - 4);

        if (path == "/stream" && streams.size() < MAX_STREAMS && send_all(fd, STREAM_HEADER))
        {
            streams.push_back(fd);
            return;
        }
        if (path == "/") send_all(fd, reply("200 OK", "text/html", PAGE));
        else if (path.size() == 6 && path.rfind("/key/", 0) == 0 && std::string("ofr").find(path[5]) != std::string::npos)
        {
            key_ = path[5];
            send_all(fd, reply("200 OK", "text/plain", "ok\n"));
        }
        else if (path == "/stream") send_all(fd, reply("503 Service Unavailable", "text/plain", "too many viewers\n"));
        else send_all(fd, reply("404 Not Found", "text/plain", "not found\n"));
        ::close(fd);
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

    int listen_fd_ = -1;
    int wake_[2] = {-1, -1}; // pipe that wakes the server thread for a new image or to stop
    std::atomic<bool> running_{true};
    std::atomic<int> key_{-1};
    std::mutex mutex_;
    cv::Mat image_;      // newest image; guarded by mutex_
    uint64_t count_ = 0; // images published; guarded by mutex_
    std::thread thread_;
};
