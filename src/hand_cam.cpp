// hand_cam.cpp
//
// Camera + hand-landmark front end: webcam -> palm detector -> landmark network (ONNX Runtime)
// -> 21 world landmarks sent over UDP to dh116_teleop (layouts in teleop_proto.h).
// The controller's reply is drawn in the preview window.
//
// Dependencies: OpenCV (core, imgproc, videoio, highgui; no highgui with HAND_CAM_HEADLESS),
// ONNX Runtime C++, Linux UDP sockets.

#include <onnxruntime_cxx_api.h>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include "teleop_proto.h"

constexpr double REPLY_TIMEOUT = 0.5;

constexpr int PALM_SIZE = 192;
constexpr int LM_SIZE   = 224;

constexpr double ROI_SCALE = 2.9;
constexpr double ROI_SHIFT = 0.5;

constexpr const char* WINDOW_NAME = "DH116 Teleop";

// HAND_CAM_HEADLESS builds link an OpenCV without highgui and always run with --no-preview.
namespace gui
{
#ifdef HAND_CAM_HEADLESS
inline void show(const char*, const cv::Mat&) {}
inline int wait_key(int) { return -1; }
inline bool visible(const char*) { return true; }
inline void close_all() {}
#else
inline void show(const char* name, const cv::Mat& image) { cv::imshow(name, image); }
inline int wait_key(int ms) { return cv::waitKey(ms); }
inline bool visible(const char* name) { return cv::getWindowProperty(name, cv::WND_PROP_VISIBLE) >= 1; }
inline void close_all() { cv::destroyAllWindows(); }
#endif
}  // namespace gui

const std::array<std::string, 6> LABELS = {
    "Thumb sweep", "Thumb bend", "Index",
    "Middle", "Ring", "Little"
};

const std::array<std::pair<int, int>, 21> CONNECTIONS = {{
    {0,1}, {1,2}, {2,3}, {3,4},
    {0,5}, {5,6}, {6,7}, {7,8},
    {5,9}, {9,10}, {10,11}, {11,12},
    {9,13}, {13,14}, {14,15}, {15,16},
    {13,17}, {17,18}, {18,19}, {19,20},
    {0,17}
}};

constexpr const char* HELP =
    "O: open hand   F: fist   R: reset   Q/Esc: quit";

std::atomic<bool> g_interrupted{false};

void signal_handler(int)
{
    g_interrupted.store(true);
}

// -----------------------------------------------------------------------------
// Time
// -----------------------------------------------------------------------------

double monotonic_seconds()
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(
        clock::now().time_since_epoch()).count();
}


// -----------------------------------------------------------------------------
// Packets
// -----------------------------------------------------------------------------

tp_landmarks make_packet(
    uint32_t seq,
    double time_s,
    bool right,
    const std::array<float, 63>* world,
    uint8_t calibration,
    bool quit)
{
    tp_landmarks packet{};

    packet.magic = TP_MAGIC_LANDMARKS;
    packet.version = TP_VERSION;
    packet.flags = (world ? TP_FLAG_DETECTED : 0) | (quit ? TP_FLAG_QUIT : 0);
    packet.cal_cmd = calibration;
    packet.seq = seq;
    packet.right_hand = right ? 1 : 0;
    packet.time_s = time_s;

    if (world)
        std::memcpy(packet.world, world->data(), sizeof(packet.world));

    return packet;
}

// -----------------------------------------------------------------------------
// CLI
// -----------------------------------------------------------------------------

struct Options
{
    std::string camera = "0";
    int width = 320;
    int height = 240;
    std::string hand = "left";
    int port = TP_PORT;

    // Holds hand_palm.onnx and hand_landmark_detector.onnx.
    std::string model_dir = "models";

    bool flip_input = false;
    bool no_preview = false;

    int frames = 0;
    int threads = 8;
    int det_every = 30;
};

void print_usage(const char* argv0)
{
    std::cout
        << "Usage: " << argv0 << " [options]\n\n"
        << "Options:\n"
        << "  --camera SOURCE       Webcam index or video file (default 0)\n"
        << "  --width N             Camera width (default 320)\n"
        << "  --height N            Camera height (default 240)\n"
        << "  --hand left|right     Hand to track (default left)\n"
        << "  --port N              Controller UDP port (default 5006)\n"
        << "  --model-dir DIR       Directory with the two ONNX models (default models)\n"
        << "  --flip-input          Horizontally flip incoming frames\n"
        << "  --no-preview          Disable the GUI\n"
        << "  --frames N            Stop after N frames; 0 means unlimited\n"
        << "  --threads N           ONNX intra-op threads (default 8)\n"
        << "  --det-every N         Palm detection interval (default 30)\n"
        << "  --help                Show this help\n";
}

Options parse_args(int argc, char** argv)
{
    Options opts;

    auto value = [&](int& i, const std::string& option) -> std::string
    {
        if (i + 1 >= argc)
            throw std::runtime_error("Missing value for " + option);

        return argv[++i];
    };

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];

        if (arg == "--help" || arg == "-h")
        {
            print_usage(argv[0]);
            std::exit(0);
        }
        else if (arg == "--camera")
            opts.camera = value(i, arg);
        else if (arg == "--width")
            opts.width = std::stoi(value(i, arg));
        else if (arg == "--height")
            opts.height = std::stoi(value(i, arg));
        else if (arg == "--hand")
            opts.hand = value(i, arg);
        else if (arg == "--port")
            opts.port = std::stoi(value(i, arg));
        else if (arg == "--model-dir")
            opts.model_dir = value(i, arg);
        else if (arg == "--flip-input")
            opts.flip_input = true;
        else if (arg == "--no-preview")
            opts.no_preview = true;
        else if (arg == "--frames")
            opts.frames = std::stoi(value(i, arg));
        else if (arg == "--threads")
            opts.threads = std::stoi(value(i, arg));
        else if (arg == "--det-every")
            opts.det_every = std::stoi(value(i, arg));
        else
            throw std::runtime_error("Unknown argument: " + arg);
    }

    if (opts.frames < 0)
        throw std::runtime_error("--frames must be nonnegative");

    if (opts.port < 1 || opts.port > 65535)
        throw std::runtime_error("--port must be between 1 and 65535");

    if (opts.width <= 0 || opts.height <= 0)
        throw std::runtime_error("--width and --height must be positive");

    if (opts.threads <= 0)
        throw std::runtime_error("--threads must be positive");

    if (opts.det_every <= 0)
        throw std::runtime_error("--det-every must be positive");

    if (opts.hand != "left" && opts.hand != "right")
        throw std::runtime_error("--hand must be left or right");

    return opts;
}

bool is_numeric(const std::string& value)
{
    return !value.empty() &&
           std::all_of(value.begin(), value.end(),
                       [](unsigned char c) { return c >= '0' && c <= '9'; });
}

// -----------------------------------------------------------------------------
// Camera setup
// -----------------------------------------------------------------------------

cv::VideoCapture open_camera(
    const std::string& source,
    int width,
    int height)
{
    cv::VideoCapture capture;

    if (is_numeric(source))
    {
        const int index = std::stoi(source);

#ifdef __linux__
        capture.open(index, cv::CAP_V4L2);
#else
        capture.open(index, cv::CAP_ANY);
#endif

        if (capture.isOpened())
        {
            capture.set(
                cv::CAP_PROP_FOURCC,
                cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

            capture.set(cv::CAP_PROP_FRAME_WIDTH, width);
            capture.set(cv::CAP_PROP_FRAME_HEIGHT, height);
            capture.set(cv::CAP_PROP_FPS, 30);
        }
    }
    else
    {
        capture.open(source);
    }

    if (!capture.isOpened())
        throw std::runtime_error("Could not open camera or video: " + source);

    return capture;
}

// -----------------------------------------------------------------------------
// Background camera reader
// Keeps only the newest frame. This allows camera capture to overlap inference.
// -----------------------------------------------------------------------------

class Grabber
{
public:
    explicit Grabber(cv::VideoCapture& capture) : capture_(capture) {}

    void start()
    {
        worker_ = std::thread(&Grabber::run, this);
    }

    void stop()
    {
        running_.store(false);

        if (worker_.joinable())
            worker_.join();

        cond_.notify_all();
    }

    ~Grabber()
    {
        stop();
    }

    struct Frame
    {
        cv::Mat image;
        double stamp = 0.0;
        uint64_t count = 0;
    };

    Frame next(uint64_t last_count, double timeout_seconds = 1.0)
    {
        std::unique_lock<std::mutex> lock(mutex_);

        const auto timeout = std::chrono::duration<double>(timeout_seconds);

        cond_.wait_for(lock, timeout, [&]
        {
            return count_ != last_count || failures_ > 30 ||
                   !running_.load();
        });

        if (failures_ > 30)
            throw std::runtime_error("The camera stopped delivering frames.");

        return Frame{frame_, stamp_, count_};
    }

private:
    void run()
    {
        while (running_.load())
        {
            cv::Mat image;
            const bool ok = capture_.read(image);
            const double now = monotonic_seconds();

            {
                std::lock_guard<std::mutex> lock(mutex_);

                if (ok)
                {
                    frame_ = std::move(image);
                    stamp_ = now;
                    ++count_;
                    failures_ = 0;
                }
                else
                {
                    ++failures_;
                }
            }

            cond_.notify_all();

            if (!ok)
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    cv::VideoCapture& capture_;

    std::thread worker_;
    std::atomic<bool> running_{true};

    std::mutex mutex_;
    std::condition_variable cond_;

    cv::Mat frame_;
    double stamp_ = 0.0;
    uint64_t count_ = 0;
    int failures_ = 0;
};


// -----------------------------------------------------------------------------
// Hand landmark data
// -----------------------------------------------------------------------------

struct ROI
{
    double cx = 0.0;
    double cy = 0.0;
    double size = 0.0;
    double rotation = 0.0;
};

struct DetectedHand
{
    std::array<cv::Point2f, 21> screen;  // normalised image coordinates
    std::array<float, 63> world;         // metres, wrist-centred, xyz per point

    bool right = false;
    float handedness_score = 0.0f;
};

// -----------------------------------------------------------------------------
// ONNX inference helper
// -----------------------------------------------------------------------------

struct TensorOutput
{
    std::vector<int64_t> shape;
    std::vector<float> values;
};

class HandLandmarker
{
public:
    HandLandmarker(
        const std::string& palm_path,
        const std::string& landmark_path,
        int num_hands = 2,
        float threshold = 0.5f,
        int threads = 4,
        int det_every = 15)
        : env_(ORT_LOGGING_LEVEL_WARNING, "DH116Hand"),
          palm_options_(),
          landmark_options_(),
          num_hands_(num_hands),
          threshold_(threshold),
          det_every_(det_every)
    {
        if (threads <= 0)
            throw std::runtime_error("ONNX thread count must be positive");

        if (det_every_ <= 0)
            throw std::runtime_error("det_every must be positive");

        palm_options_.SetIntraOpNumThreads(threads);
        palm_options_.SetInterOpNumThreads(1);

        landmark_options_.SetIntraOpNumThreads(threads);
        landmark_options_.SetInterOpNumThreads(1);

        palm_ = Ort::Session(env_, palm_path.c_str(), palm_options_);
        landmark_ = Ort::Session(env_, landmark_path.c_str(),
                                 landmark_options_);

        palm_input_name_ = get_input_name(palm_, 0);
        landmark_input_name_ = get_input_name(landmark_, 0);

        palm_output_names_ = get_output_names(palm_);
        landmark_output_names_ = get_output_names(landmark_);
    }

    // want_right: once that hand is found, only it is tracked (saves a landmark pass per frame).
    std::vector<DetectedHand> detect(const cv::Mat& rgb, bool want_right)
    {
        if (rgb.empty() || rgb.type() != CV_8UC3)
            throw std::runtime_error("Expected a nonempty RGB uint8 image");

        const int h = rgb.rows;
        const int w = rgb.cols;

        ++det_counter_;

        const bool run_palm =
            tracked_.empty() || (det_counter_ % det_every_ == 0);

        if (run_palm)
        {
            int pad_x = 0;
            int pad_y = 0;
            double scale = 0.0;

            const cv::Mat palm_input = make_palm_input(
                rgb, pad_x, pad_y, scale);

            const auto palm_outputs = run_model(
                palm_, palm_input_name_, palm_output_names_,
                palm_input, true);

            tracked_ = rois_from_palm_dets(
                palm_outputs.at(0), pad_x, pad_y, scale);
        }

        std::vector<DetectedHand> result;
        std::vector<ROI> next_tracked;
        std::vector<ROI> wanted_tracked;

        for (const ROI& roi : tracked_)
        {
            cv::Mat affine;
            cv::Mat crop = make_landmark_crop(rgb, roi, affine);

            const auto lm_outputs = run_model(
                landmark_, landmark_input_name_,
                landmark_output_names_, crop, false);

            if (lm_outputs.size() < 4 ||
                lm_outputs[0].values.size() < 63 ||
                lm_outputs[1].values.empty() ||
                lm_outputs[2].values.empty() ||
                lm_outputs[3].values.size() < 63)
                throw std::runtime_error(
                    "Landmark model must return 63 screen values, presence, "
                    "handedness and 63 world values");

            if (lm_outputs[1].values[0] < threshold_)
                continue;

            const float raw_hand = lm_outputs[2].values[0];

            DetectedHand hand;
            hand.right = raw_hand > 0.5f;
            hand.handedness_score =
                raw_hand > 0.5f ? raw_hand : 1.0f - raw_hand;

            const auto& pts = lm_outputs[0].values;

            cv::Mat inverse;
            cv::invertAffineTransform(affine, inverse);

            std::array<cv::Point2f, 21> xy_orig{};

            for (int i = 0; i < 21; ++i)
            {
                const double x = pts[i * 3 + 0];
                const double y = pts[i * 3 + 1];

                xy_orig[i].x = static_cast<float>(
                    inverse.at<double>(0, 0) * x +
                    inverse.at<double>(0, 1) * y +
                    inverse.at<double>(0, 2));

                xy_orig[i].y = static_cast<float>(
                    inverse.at<double>(1, 0) * x +
                    inverse.at<double>(1, 1) * y +
                    inverse.at<double>(1, 2));

                hand.screen[i] = {
                    xy_orig[i].x / static_cast<float>(w),
                    xy_orig[i].y / static_cast<float>(h)
                };
            }

            std::copy_n(lm_outputs[3].values.begin(), 63, hand.world.begin());

            result.push_back(hand);
            next_tracked.push_back(roi_from_landmarks(xy_orig));

            if (hand.right == want_right)
                wanted_tracked.push_back(next_tracked.back());
        }

        tracked_ = wanted_tracked.empty() ? next_tracked : wanted_tracked;

        return result;
    }

private:
    static std::string get_input_name(
        Ort::Session& session,
        size_t index)
    {
        Ort::AllocatorWithDefaultOptions allocator;
        auto name = session.GetInputNameAllocated(index, allocator);
        return name.get();
    }

    static std::vector<std::string> get_output_names(
        Ort::Session& session)
    {
        Ort::AllocatorWithDefaultOptions allocator;
        const size_t count = session.GetOutputCount();

        std::vector<std::string> names;
        names.reserve(count);

        for (size_t i = 0; i < count; ++i)
        {
            auto name = session.GetOutputNameAllocated(i, allocator);
            names.emplace_back(name.get());
        }

        return names;
    }

    static std::vector<const char*> c_string_pointers(
        const std::vector<std::string>& strings)
    {
        std::vector<const char*> result;
        result.reserve(strings.size());

        for (const auto& s : strings)
            result.push_back(s.c_str());

        return result;
    }

    // Converts to float in [0,1] into the reused input_ buffer. NHWC is one convertTo;
    // NCHW converts and then splits the channels straight into input_.
    void fill_input(const cv::Mat& rgb, bool nchw)
    {
        const int rows = rgb.rows;
        const int cols = rgb.cols;

        input_.resize(static_cast<size_t>(rows) * cols * 3);

        constexpr double inv255 = 1.0 / 255.0;

        if (!nchw)
        {
            cv::Mat dst(rows, cols, CV_32FC3, input_.data());
            rgb.convertTo(dst, CV_32FC3, inv255);
            return;
        }

        rgb.convertTo(scratch_, CV_32FC3, inv255);

        const size_t plane = static_cast<size_t>(rows) * cols;
        cv::Mat planes[3] = {
            cv::Mat(rows, cols, CV_32F, input_.data()),
            cv::Mat(rows, cols, CV_32F, input_.data() + plane),
            cv::Mat(rows, cols, CV_32F, input_.data() + 2 * plane)};

        cv::split(scratch_, planes);
    }

    static cv::Mat make_palm_input(
        const cv::Mat& rgb,
        int& pad_x,
        int& pad_y,
        double& scale)
    {
        const int h = rgb.rows;
        const int w = rgb.cols;

        scale = static_cast<double>(PALM_SIZE) / std::max(h, w);

        const int nh = static_cast<int>(h * scale);
        const int nw = static_cast<int>(w * scale);

        pad_y = (PALM_SIZE - nh) / 2;
        pad_x = (PALM_SIZE - nw) / 2;

        cv::Mat resized;
        cv::resize(rgb, resized, cv::Size(nw, nh));

        cv::Mat padded = cv::Mat::zeros(
            PALM_SIZE, PALM_SIZE, CV_8UC3);

        resized.copyTo(
            padded(cv::Rect(pad_x, pad_y, nw, nh)));

        return padded;
    }

    std::vector<TensorOutput> run_model(
        Ort::Session& session,
        const std::string& input_name,
        const std::vector<std::string>& output_names,
        const cv::Mat& rgb,
        bool nchw)
    {
        fill_input(rgb, nchw);
        std::vector<float>& input_values = input_;

        std::vector<int64_t> shape;

        if (nchw)
        {
            shape = {
                1, 3,
                static_cast<int64_t>(rgb.rows),
                static_cast<int64_t>(rgb.cols)
            };
        }
        else
        {
            shape = {
                1,
                static_cast<int64_t>(rgb.rows),
                static_cast<int64_t>(rgb.cols),
                3
            };
        }

        auto memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_values.data(),
            input_values.size(),
            shape.data(),
            shape.size());

        const char* input_names[] = {input_name.c_str()};
        const auto output_name_ptrs = c_string_pointers(output_names);

        auto outputs = session.Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1,
            output_name_ptrs.data(),
            output_name_ptrs.size());

        std::vector<TensorOutput> result;
        result.reserve(outputs.size());

        for (auto& output : outputs)
        {
            if (!output.IsTensor())
                throw std::runtime_error("ONNX output is not a tensor");

            auto info = output.GetTensorTypeAndShapeInfo();

            TensorOutput item;
            item.shape = info.GetShape();

            if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
                throw std::runtime_error(
                    "Expected float32 ONNX model outputs");

            const size_t count = info.GetElementCount();
            const float* data = output.GetTensorData<float>();

            item.values.assign(data, data + count);
            result.push_back(std::move(item));
        }

        return result;
    }

    std::vector<ROI> rois_from_palm_dets(
        const TensorOutput& output,
        int pad_x,
        int pad_y,
        double scale)
    {
        std::vector<ROI> rois;

        // Expected final dimension:
        // [score, cx, cy, box_size, kp0_x, kp0_y, kp2_x, kp2_y]
        if (output.shape.empty())
            return rois;

        const int64_t row_width = output.shape.back();

        if (row_width < 8 ||
            output.values.size() % static_cast<size_t>(row_width) != 0)
        {
            throw std::runtime_error(
                "Unexpected palm model output shape");
        }

        struct Detection
        {
            float score;
            const float* row;
        };

        std::vector<Detection> detections;

        const size_t rows =
            output.values.size() / static_cast<size_t>(row_width);

        for (size_t i = 0; i < rows; ++i)
        {
            const float* row =
                output.values.data() + i * row_width;

            if (row[0] >= threshold_)
                detections.push_back({row[0], row});
        }

        std::sort(
            detections.begin(),
            detections.end(),
            [](const Detection& a, const Detection& b)
            {
                return a.score > b.score;
            });

        if (detections.size() > static_cast<size_t>(num_hands_))
            detections.resize(num_hands_);

        for (const auto& detection : detections)
        {
            const float* det = detection.row;

            const double cx = det[1];
            const double cy = det[2];
            const double sz = det[3];

            const double k0x = det[4];
            const double k0y = det[5];
            const double k2x = det[6];
            const double k2y = det[7];

            const double cx_px = (cx * PALM_SIZE - pad_x) / scale;
            const double cy_px = (cy * PALM_SIZE - pad_y) / scale;

            const double k0x_px = (k0x * PALM_SIZE - pad_x) / scale;
            const double k0y_px = (k0y * PALM_SIZE - pad_y) / scale;

            const double k2x_px = (k2x * PALM_SIZE - pad_x) / scale;
            const double k2y_px = (k2y * PALM_SIZE - pad_y) / scale;

            const double sz_px = sz * PALM_SIZE / scale;

            const double rotation =
                0.5 * M_PI -
                std::atan2(
                    -(k2y_px - k0y_px),
                    k2x_px - k0x_px);

            const double roi_sz = ROI_SCALE * sz_px;

            const double roi_cx =
                cx_px + ROI_SHIFT * sz_px * std::sin(rotation);

            const double roi_cy =
                cy_px - ROI_SHIFT * sz_px * std::cos(rotation);

            rois.push_back({roi_cx, roi_cy, roi_sz, rotation});
        }

        return rois;
    }

    static cv::Mat make_landmark_crop(
        const cv::Mat& rgb,
        const ROI& roi,
        cv::Mat& affine)
    {
        const double sf = LM_SIZE / roi.size;

        const double cr = std::cos(-roi.rotation);
        const double sr = std::sin(-roi.rotation);

        // CV_64F is used so that the inverse transformation retains
        // double-precision coefficients.
        affine = (cv::Mat_<double>(2, 3) <<
            sf * cr,
            -sf * sr,
            LM_SIZE / 2.0 -
                sf * (cr * roi.cx - sr * roi.cy),

            sf * sr,
            sf * cr,
            LM_SIZE / 2.0 -
                sf * (sr * roi.cx + cr * roi.cy));

        cv::Mat crop;

        cv::warpAffine(
            rgb,
            crop,
            affine,
            cv::Size(LM_SIZE, LM_SIZE));

        return crop;
    }

    static ROI roi_from_landmarks(
        const std::array<cv::Point2f, 21>& points)
    {
        const double dx = points[9].x - points[0].x;
        const double dy = points[9].y - points[0].y;

        const double rotation =
            0.5 * M_PI - std::atan2(-dy, dx);

        double min_x = points[0].x;
        double max_x = points[0].x;
        double min_y = points[0].y;
        double max_y = points[0].y;

        for (const auto& p : points)
        {
            min_x = std::min(min_x, static_cast<double>(p.x));
            max_x = std::max(max_x, static_cast<double>(p.x));
            min_y = std::min(min_y, static_cast<double>(p.y));
            max_y = std::max(max_y, static_cast<double>(p.y));
        }

        const double cx = (min_x + max_x) * 0.5;
        const double cy = (min_y + max_y) * 0.5;

        const double diagonal = std::hypot(max_x - min_x, max_y - min_y);

        return {cx, cy, diagonal * 1.2, rotation};
    }

    Ort::Env env_;
    Ort::SessionOptions palm_options_;
    Ort::SessionOptions landmark_options_;

    Ort::Session palm_{nullptr};
    Ort::Session landmark_{nullptr};

    std::string palm_input_name_;
    std::string landmark_input_name_;

    std::vector<std::string> palm_output_names_;
    std::vector<std::string> landmark_output_names_;

    int num_hands_ = 2;
    float threshold_ = 0.5f;

    std::vector<float> input_;  // reused model input buffer
    cv::Mat scratch_;           // reused float image for the NCHW path
    int det_every_ = 15;
    uint64_t det_counter_ = 0;

    std::vector<ROI> tracked_;
};

// -----------------------------------------------------------------------------
// UDP socket
// -----------------------------------------------------------------------------

class UdpClient
{
public:
    explicit UdpClient(int port)
    {
        fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);

        if (fd_ < 0)
            throw std::runtime_error("Could not create UDP socket");

        const int flags = fcntl(fd_, F_GETFL, 0);

        if (flags < 0 || fcntl(fd_, F_SETFL, flags | O_NONBLOCK) < 0)
        {
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("Could not make UDP socket non-blocking");
        }

        target_ = {};
        target_.sin_family = AF_INET;
        target_.sin_port = htons(static_cast<uint16_t>(port));

        if (inet_pton(AF_INET, "127.0.0.1", &target_.sin_addr) != 1)
        {
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("Could not configure UDP destination");
        }
    }

    ~UdpClient()
    {
        close();
    }

    void send(const tp_landmarks& packet)
    {
        const ssize_t sent = sendto(
            fd_,
            &packet,
            sizeof(packet),
            0,
            reinterpret_cast<const sockaddr*>(&target_),
            sizeof(target_));

        if (sent < 0)
            throw std::runtime_error(
                std::string("UDP send failed: ") + std::strerror(errno));

        if (static_cast<size_t>(sent) != sizeof(packet))
            throw std::runtime_error("UDP packet was not sent completely");
    }

    std::vector<tp_status> receive_all()
    {
        std::vector<tp_status> results;
        tp_status buffer;

        while (true)
        {
            const ssize_t received = recvfrom(
                fd_, &buffer, sizeof(buffer), 0, nullptr, nullptr);

            if (received < 0)
            {
                if (errno == EINTR)
                    continue;

                break;
            }

            if (static_cast<size_t>(received) == sizeof(buffer) &&
                buffer.magic == TP_MAGIC_STATUS &&
                buffer.version == TP_VERSION)
            {
                buffer.msg[sizeof(buffer.msg) - 1] = '\0';
                results.push_back(buffer);
            }
        }

        return results;
    }

    void close()
    {
        if (fd_ >= 0)
        {
            ::close(fd_);
            fd_ = -1;
        }
    }

private:
    int fd_ = -1;
    sockaddr_in target_{};
};

// -----------------------------------------------------------------------------
// Preview drawing
// -----------------------------------------------------------------------------

cv::Mat draw_preview(
    cv::Mat image,
    const std::vector<DetectedHand>& hands,
    int selected,
    const std::optional<tp_status>& status,
    const std::vector<std::string>& lines)
{
    const int height = image.rows;
    const int width = image.cols;

    for (size_t index = 0; index < hands.size(); ++index)
    {
        const auto& landmarks = hands[index].screen;

        std::array<cv::Point, 21> points;

        for (int i = 0; i < 21; ++i)
        {
            points[i] = cv::Point(
                static_cast<int>(landmarks[i].x * width),
                static_cast<int>(landmarks[i].y * height));
        }

        const cv::Scalar color =
            static_cast<int>(index) == selected
                ? cv::Scalar(80, 220, 80)
                : cv::Scalar(150, 150, 150);

        for (const auto& edge : CONNECTIONS)
        {
            cv::line(
                image,
                points[edge.first],
                points[edge.second],
                color,
                2);
        }

        for (const auto& point : points)
        {
            const cv::Scalar point_color =
                static_cast<int>(index) == selected
                    ? cv::Scalar(40, 40, 230)
                    : color;

            cv::circle(image, point, 3, point_color, -1);
        }
    }

    cv::flip(image, image, 1);

    cv::rectangle(
        image,
        cv::Point(0, 0),
        cv::Point(width, 14 + 20 * static_cast<int>(lines.size())),
        cv::Scalar(0, 0, 0),
        -1);

    cv::rectangle(
        image,
        cv::Point(0, height - 30 - 20 * static_cast<int>(LABELS.size())),
        cv::Point(272, height),
        cv::Scalar(0, 0, 0),
        -1);

    for (size_t row = 0; row < lines.size(); ++row)
    {
        cv::putText(
            image,
            lines[row],
            cv::Point(10, 22 + 20 * static_cast<int>(row)),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(255, 255, 255),
            1);
    }

    for (size_t row = 0; row < LABELS.size(); ++row)
    {
        const int y =
            height - 12 -
            20 * (static_cast<int>(LABELS.size()) - 1 -
                  static_cast<int>(row));

        cv::putText(
            image,
            LABELS[row],
            cv::Point(10, y),
            cv::FONT_HERSHEY_SIMPLEX,
            0.45,
            cv::Scalar(255, 255, 255),
            1);

        cv::rectangle(
            image,
            cv::Point(110, y - 11),
            cv::Point(260, y + 1),
            cv::Scalar(60, 60, 60),
            -1);

        if (status.has_value())
        {
            const int target_x =
                110 + static_cast<int>(
                    150.0 * status->target[row] / 10000.0);

            cv::rectangle(
                image,
                cv::Point(110, y - 11),
                cv::Point(target_x, y + 1),
                cv::Scalar(80, 200, 255),
                -1);

            if (status->have_fb)
            {
                const int measured = std::clamp<int>(
                    status->measured[row], 0, 10000);

                const int x =
                    110 + static_cast<int>(150.0 * measured / 10000.0);

                cv::line(
                    image,
                    cv::Point(x, y - 14),
                    cv::Point(x, y + 4),
                    cv::Scalar(80, 255, 80),
                    2);
            }
        }
    }

    return image;
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main(int argc, char** argv)
{
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    Options args;

    try
    {
        args = parse_args(argc, argv);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Argument error: " << e.what() << "\n";
        print_usage(argv[0]);
        return 2;
    }

#ifdef HAND_CAM_HEADLESS
    args.no_preview = true;
#endif

    const std::string model_dir = args.model_dir;
    const std::string palm_path =
        model_dir + "/hand_palm.onnx";

    const std::string landmark_path =
        model_dir + "/hand_landmark_detector.onnx";

    if (::access(palm_path.c_str(), R_OK) != 0)
    {
        std::cerr << "ONNX model not found: " << palm_path << "\n";
        return 2;
    }

    if (::access(landmark_path.c_str(), R_OK) != 0)
    {
        std::cerr << "ONNX model not found: " << landmark_path << "\n";
        return 2;
    }

    std::unique_ptr<HandLandmarker> landmarker;
    cv::VideoCapture capture;
    std::unique_ptr<Grabber> grabber;
    std::unique_ptr<UdpClient> sock;

    try
    {
        landmarker = std::make_unique<HandLandmarker>(
            palm_path,
            landmark_path,
            2,
            0.5f,
            args.threads,
            args.det_every);

        capture = open_camera(args.camera, args.width, args.height);

        const bool is_file = !is_numeric(args.camera);

        if (!is_file)
        {
            grabber = std::make_unique<Grabber>(capture);
            grabber->start();
        }

        sock = std::make_unique<UdpClient>(args.port);
    }
    catch (const std::exception& e)
    {
        std::cerr << "hand_cam error: " << e.what() << "\n";

        if (grabber)
            grabber->stop();

        capture.release();
        return 2;
    }

    const bool right = args.hand == "right";

    uint32_t seq = 0;
    int frames = 0;

    uint64_t count = 0;

    uint8_t cal = TP_CAL_NONE;

    double fps = 0.0;
    double previous = monotonic_seconds();

    std::optional<tp_status> status;
    double status_time = 0.0;

    std::string message;

    std::cout
        << "Tracking your " << args.hand
        << " hand -> dh116_teleop on 127.0.0.1:"
        << args.port << ". " << HELP << std::endl;

    int exit_code = 0;

    try
    {
        while (!g_interrupted.load() &&
               (!args.frames || frames < args.frames))
        {
            cv::Mat image;
            double stamp = 0.0;

            if (grabber)
            {
                auto frame = grabber->next(count);

                image = frame.image;
                stamp = frame.stamp;
                count = frame.count;

                if (image.empty())
                    continue;
            }
            else
            {
                const bool ok = capture.read(image);
                stamp = monotonic_seconds();

                if (!ok || image.empty())
                    break;
            }

            if (args.flip_input)
                cv::flip(image, image, 1);

            cv::Mat rgb;
            cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);

            const std::vector<DetectedHand> hands =
                landmarker->detect(rgb, right);

            int selected = -1;
            float selected_score = -1.0f;

            for (size_t i = 0; i < hands.size(); ++i)
            {
                if (hands[i].right == right &&
                    hands[i].handedness_score >= selected_score)
                {
                    selected = static_cast<int>(i);
                    selected_score = hands[i].handedness_score;
                }
            }

            const std::array<float, 63>* world =
                selected >= 0 ? &hands[selected].world : nullptr;

            try
            {
                sock->send(make_packet(
                    seq, stamp, right, world, cal, false));
            }
            catch (const std::exception& e)
            {
                message = std::string("Send failed: ") + e.what();
            }

            cal = TP_CAL_NONE;

            // Non-blocking receive: consume all queued replies.
            for (const auto& reply : sock->receive_all())
            {
                status = reply;
                status_time = monotonic_seconds();
            }

            ++seq;
            ++frames;

            const double now = monotonic_seconds();

            if (frames > 1)
            {
                fps = 0.9 * fps +
                      0.1 / std::max(now - previous, 1e-3);
            }
            else
            {
                fps = 0.0;
            }

            previous = now;

            const bool alive =
                status.has_value() &&
                (now - status_time < REPLY_TIMEOUT);

            if (alive && message != status->msg)
            {
                message = status->msg;
                std::cout << message << std::endl;
            }

            if (args.no_preview)
                continue;

            std::vector<std::string> lines;

            lines.push_back(
                args.hand == "left" ? "LEFT hand" : "RIGHT hand");

            lines[0] += "   ";

            {
                std::ostringstream stream;
                stream << std::fixed << std::setprecision(1)
                       << fps << " FPS   "
                       << (world
                           ? "Hand detected"
                           : "No " + args.hand + " hand visible");

                lines[0] += stream.str();
            }

            static const std::array<const char*, 3> STATES = {
                "DRY RUN (no EtherCAT)",
                "hand RUNNING",
                "stopping"
            };

            if (alive)
            {
                if (status->state < STATES.size())
                    lines.push_back(STATES[status->state]);
                else
                    lines.push_back("?");
            }
            else
            {
                lines.push_back(
                    "No reply from dh116_teleop on port " +
                    std::to_string(args.port));
            }

            lines.push_back(message);
            lines.push_back(HELP);

            cv::Mat preview = draw_preview(
                image,
                hands,
                selected,
                alive ? status : std::nullopt,
                lines);

            gui::show(WINDOW_NAME, preview);

            const int key = gui::wait_key(1) & 0xFF;

            if (key == 'q' || key == 27)
                break;

            if (!gui::visible(WINDOW_NAME))
                break;

            switch (key)
            {
                case 'o':
                    cal = TP_CAL_OPEN;
                    break;

                case 'f':
                    cal = TP_CAL_FIST;
                    break;

                case 'r':
                    cal = TP_CAL_RESET;
                    break;

                default:
                    cal = TP_CAL_NONE;
                    break;
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "hand_cam error: " << e.what() << "\n";
        exit_code = 2;
    }

    // Tell the controller to open the hand and stop; repeated because UDP may drop one.
    for (int i = 0; i < 3; ++i)
    {
        try
        {
            sock->send(make_packet(
                seq++, monotonic_seconds(), right, nullptr, TP_CAL_NONE, true));
        }
        catch (...)
        {
            // Continue cleanup even if the controller is unavailable.
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    if (grabber)
        grabber->stop();

    capture.release();

    if (!args.no_preview)
        gui::close_all();

    sock->close();

    return exit_code;
}
