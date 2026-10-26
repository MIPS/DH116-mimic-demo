// hand_cam.cpp
//
// Native C++ replacement for the Python ONNX hand-tracking front end.
//
// Dependencies:
//   - OpenCV: core, imgproc, imgcodecs, videoio, highgui (not needed with HAND_CAM_HEADLESS)
//   - ONNX Runtime C++
//   - Linux/POSIX UDP sockets
//
// The UDP wire format is compatible with teleop_proto.h and the
// Python struct formats:
//   LANDMARKS = "<IHBBIB3xd63f4x"  (280 bytes)
//   STATUS    = "<IHBBI6f6h6hB3x64s" (128 bytes)
//
// Build configuration is provided after the source.

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

// -----------------------------------------------------------------------------
// Protocol constants: keep synchronized with teleop_proto.h.
// -----------------------------------------------------------------------------

constexpr uint16_t DEFAULT_PORT = 5006;
constexpr uint16_t VERSION = 1;

constexpr uint32_t MAGIC_LANDMARKS = 0x4D4C4844;
constexpr uint32_t MAGIC_STATUS    = 0x54534844;

constexpr uint8_t FLAG_DETECTED = 0x01;
constexpr uint8_t FLAG_QUIT     = 0x02;

constexpr uint8_t CAL_NONE  = 0;
constexpr uint8_t CAL_OPEN  = 1;
constexpr uint8_t CAL_FIST  = 2;
constexpr uint8_t CAL_RESET = 3;

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

uint64_t monotonic_milliseconds()
{
    return static_cast<uint64_t>(monotonic_seconds() * 1000.0);
}


// -----------------------------------------------------------------------------
// Explicit little-endian packet serialization.
//
// Do not send a native C++ struct directly. Native structure padding and
// alignment must not change the established Python/C wire protocol.
// -----------------------------------------------------------------------------

constexpr size_t LANDMARK_PACKET_SIZE = 280;
constexpr size_t STATUS_PACKET_SIZE   = 128;

using LandmarkPacket = std::array<uint8_t, LANDMARK_PACKET_SIZE>;
using StatusPacket   = std::array<uint8_t, STATUS_PACKET_SIZE>;

void put_u16_le(uint8_t* dst, uint16_t value)
{
    dst[0] = static_cast<uint8_t>(value);
    dst[1] = static_cast<uint8_t>(value >> 8);
}

void put_u32_le(uint8_t* dst, uint32_t value)
{
    for (int i = 0; i < 4; ++i)
        dst[i] = static_cast<uint8_t>(value >> (8 * i));
}

void put_u64_le(uint8_t* dst, uint64_t value)
{
    for (int i = 0; i < 8; ++i)
        dst[i] = static_cast<uint8_t>(value >> (8 * i));
}

uint16_t get_u16_le(const uint8_t* src)
{
    return static_cast<uint16_t>(src[0]) |
           (static_cast<uint16_t>(src[1]) << 8);
}

uint32_t get_u32_le(const uint8_t* src)
{
    uint32_t result = 0;
    for (int i = 0; i < 4; ++i)
        result |= static_cast<uint32_t>(src[i]) << (8 * i);
    return result;
}

uint64_t get_u64_le(const uint8_t* src)
{
    uint64_t result = 0;
    for (int i = 0; i < 8; ++i)
        result |= static_cast<uint64_t>(src[i]) << (8 * i);
    return result;
}

void put_float_le(uint8_t* dst, float value)
{
    uint32_t bits;
    static_assert(sizeof(float) == sizeof(uint32_t));
    std::memcpy(&bits, &value, sizeof(bits));
    put_u32_le(dst, bits);
}

float get_float_le(const uint8_t* src)
{
    uint32_t bits = get_u32_le(src);
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void put_double_le(uint8_t* dst, double value)
{
    uint64_t bits;
    static_assert(sizeof(double) == sizeof(uint64_t));
    std::memcpy(&bits, &value, sizeof(bits));
    put_u64_le(dst, bits);
}

double get_double_le(const uint8_t* src)
{
    uint64_t bits = get_u64_le(src);
    double value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void put_i16_le(uint8_t* dst, int16_t value)
{
    uint16_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    put_u16_le(dst, bits);
}

int16_t get_i16_le(const uint8_t* src)
{
    uint16_t bits = get_u16_le(src);
    int16_t value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

LandmarkPacket encode_packet(
    uint32_t seq,
    double time_s,
    bool right,
    const std::optional<std::array<float, 63>>& world,
    uint8_t calibration,
    bool quit)
{
    LandmarkPacket packet{};

    uint8_t flags = 0;

    if (world.has_value())
        flags |= FLAG_DETECTED;

    if (quit)
        flags |= FLAG_QUIT;

    // "<IHBBIB3xd63f4x"
    //
    // Offset  Size  Field
    // 0       4     magic
    // 4       2     version
    // 6       1     flags
    // 7       1     calibration
    // 8       4     sequence
    // 12      1     selected hand is right
    // 13      3     padding
    // 16      8     monotonic timestamp in seconds
    // 24      252   63 float coordinates
    // 276     4     trailing padding

    put_u32_le(packet.data() + 0, MAGIC_LANDMARKS);
    put_u16_le(packet.data() + 4, VERSION);

    packet[6] = flags;
    packet[7] = calibration;

    put_u32_le(packet.data() + 8, seq);
    packet[12] = right ? 1 : 0;

    put_double_le(packet.data() + 16, time_s);

    if (world.has_value())
    {
        for (size_t i = 0; i < 63; ++i)
            put_float_le(packet.data() + 24 + i * 4, (*world)[i]);
    }

    return packet;
}

struct Status
{
    uint8_t state = 0;
    uint8_t valid = 0;
    uint32_t seq = 0;

    std::array<float, 6> closure{};
    std::array<int16_t, 6> target{};
    std::array<int16_t, 6> measured{};

    uint8_t have_fb = 0;
    std::string message;
};

std::optional<Status> decode_status(
    const uint8_t* packet,
    size_t length)
{
    if (length != STATUS_PACKET_SIZE)
        return std::nullopt;

    if (get_u32_le(packet + 0) != MAGIC_STATUS ||
        get_u16_le(packet + 4) != VERSION)
        return std::nullopt;

    Status result;

    // "<IHBBI6f6h6hB3x64s"
    result.state = packet[6];
    result.valid = packet[7];
    result.seq = get_u32_le(packet + 8);

    for (size_t i = 0; i < 6; ++i)
        result.closure[i] = get_float_le(packet + 12 + i * 4);

    for (size_t i = 0; i < 6; ++i)
        result.target[i] = get_i16_le(packet + 36 + i * 2);

    for (size_t i = 0; i < 6; ++i)
        result.measured[i] = get_i16_le(packet + 48 + i * 2);

    result.have_fb = packet[60];

    const char* msg = reinterpret_cast<const char*>(packet + 64);
    size_t msg_len = 0;

    while (msg_len < 64 && msg[msg_len] != '\0')
        ++msg_len;

    result.message.assign(msg, msg_len);

    return result;
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
    int port = DEFAULT_PORT;

    // The model option is used to derive the two ONNX paths from its directory.
    // This default assumes the application is launched from the project root.
    std::string model = "models/hand_landmarker.task";

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
        << "  --model PATH          Path used to derive ONNX model paths\n"
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
        else if (arg == "--model")
            opts.model = value(i, arg);
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

struct Landmark
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct ROI
{
    double cx = 0.0;
    double cy = 0.0;
    double size = 0.0;
    double rotation = 0.0;
};

struct DetectedHand
{
    std::array<Landmark, 21> screen;
    std::array<Landmark, 21> world;

    std::string side;
    float handedness_score = 0.0f;
};

struct DetectionResult
{
    std::vector<DetectedHand> hands;
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

        auto lm_info = landmark_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
        const auto lm_shape = lm_info.GetShape();

        landmark_nchw_ =
            lm_shape.size() == 4 && lm_shape[1] == 3;

        has_world_ = landmark_output_names_.size() >= 4;

        padded_ = cv::Mat::zeros(PALM_SIZE, PALM_SIZE, CV_8UC3);
    }

    DetectionResult detect(const cv::Mat& rgb)
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

        DetectionResult result;
        std::vector<ROI> next_tracked;

        for (const ROI& roi : tracked_)
        {
            cv::Mat affine;
            cv::Mat crop = make_landmark_crop(rgb, roi, affine);

            const auto lm_outputs = run_model(
                landmark_, landmark_input_name_,
                landmark_output_names_, crop, landmark_nchw_);

            if (lm_outputs.size() < 3)
                throw std::runtime_error(
                    "Landmark model must return screen landmarks, "
                    "presence, and handedness");

            if (lm_outputs[1].values.empty() ||
                lm_outputs[2].values.empty())
                throw std::runtime_error("Empty landmark model output");

            if (lm_outputs[1].values[0] < threshold_)
                continue;

            const float raw_hand = lm_outputs[2].values[0];

            DetectedHand hand;
            hand.side = raw_hand > 0.5f ? "Right" : "Left";
            hand.handedness_score =
                raw_hand > 0.5f ? raw_hand : 1.0f - raw_hand;

            const auto& pts = lm_outputs[0].values;

            if (pts.size() < 63)
                throw std::runtime_error(
                    "Screen landmark output has fewer than 63 values");

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
                    xy_orig[i].y / static_cast<float>(h),
                    pts[i * 3 + 2] / static_cast<float>(LM_SIZE)
                };
            }

            if (has_world_ && lm_outputs.size() >= 4 &&
                lm_outputs[3].values.size() >= 63)
            {
                const auto& world = lm_outputs[3].values;

                for (int i = 0; i < 21; ++i)
                {
                    hand.world[i] = {
                        world[i * 3 + 0],
                        world[i * 3 + 1],
                        world[i * 3 + 2]
                    };
                }
            }
            else
            {
                for (int i = 0; i < 21; ++i)
                {
                    hand.world[i] = {
                        hand.screen[i].x - hand.screen[0].x,
                        hand.screen[i].y - hand.screen[0].y,
                        0.0f
                    };
                }
            }

            result.hands.push_back(hand);
            next_tracked.push_back(roi_from_landmarks(xy_orig));
        }

        tracked_ = next_tracked;

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

    static std::vector<float> to_chw_float(const cv::Mat& rgb)
    {
        std::vector<float> values(
            static_cast<size_t>(rgb.rows) * rgb.cols * 3);

        size_t index = 0;

        for (int c = 0; c < 3; ++c)
        {
            for (int y = 0; y < rgb.rows; ++y)
            {
                const auto* row = rgb.ptr<cv::Vec3b>(y);

                for (int x = 0; x < rgb.cols; ++x)
                    values[index++] = row[x][c] / 255.0f;
            }
        }

        return values;
    }

    static std::vector<float> to_hwc_float(const cv::Mat& rgb)
    {
        std::vector<float> values(
            static_cast<size_t>(rgb.rows) * rgb.cols * 3);

        size_t index = 0;

        for (int y = 0; y < rgb.rows; ++y)
        {
            const auto* row = rgb.ptr<cv::Vec3b>(y);

            for (int x = 0; x < rgb.cols; ++x)
            {
                values[index++] = row[x][0] / 255.0f;
                values[index++] = row[x][1] / 255.0f;
                values[index++] = row[x][2] / 255.0f;
            }
        }

        return values;
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
        std::vector<float> input_values =
            nchw ? to_chw_float(rgb) : to_hwc_float(rgb);

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

    bool landmark_nchw_ = false;
    bool has_world_ = true;

    int det_every_ = 15;
    uint64_t det_counter_ = 0;

    cv::Mat padded_;
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

    void send(const LandmarkPacket& packet)
    {
        const ssize_t sent = sendto(
            fd_,
            packet.data(),
            packet.size(),
            0,
            reinterpret_cast<const sockaddr*>(&target_),
            sizeof(target_));

        if (sent < 0)
            throw std::runtime_error(
                std::string("UDP send failed: ") + std::strerror(errno));

        if (static_cast<size_t>(sent) != packet.size())
            throw std::runtime_error("UDP packet was not sent completely");
    }

    std::vector<Status> receive_all()
    {
        std::vector<Status> results;
        uint8_t buffer[512];

        while (true)
        {
            const ssize_t received = recvfrom(
                fd_, buffer, sizeof(buffer), 0, nullptr, nullptr);

            if (received < 0)
            {
                if (errno == EAGAIN || errno == EWOULDBLOCK)
                    break;

                if (errno == EINTR)
                    continue;

                break;
            }

            auto status = decode_status(
                buffer, static_cast<size_t>(received));

            if (status.has_value())
                results.push_back(*status);
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
    const std::optional<Status>& status,
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

    // Match Python:
    // palm_onnx = args.model.parent / "hand_palm.onnx"
    // lm_onnx   = args.model.parent / "hand_landmark_detector.onnx"

    const std::string model_dir = [&]()
    {
        const size_t pos = args.model.find_last_of('/');
        return pos == std::string::npos
            ? std::string(".")
            : args.model.substr(0, pos);
    }();

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
            1,
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

    uint8_t cal = CAL_NONE;

    double fps = 0.0;
    double previous = monotonic_seconds();

    std::optional<Status> status;
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

            DetectionResult result = landmarker->detect(rgb);

            int selected = -1;
            float selected_score = -1.0f;

            for (size_t i = 0; i < result.hands.size(); ++i)
            {
                const auto& hand = result.hands[i];

                std::string side = hand.side;

                std::transform(
                    side.begin(),
                    side.end(),
                    side.begin(),
                    [](unsigned char c)
                    {
                        return static_cast<char>(std::tolower(c));
                    });

                if (side == args.hand &&
                    hand.handedness_score >= selected_score)
                {
                    selected = static_cast<int>(i);
                    selected_score = hand.handedness_score;
                }
            }

            std::optional<std::array<float, 63>> world;

            if (selected >= 0)
            {
                std::array<float, 63> coordinates{};

                const auto& landmarks =
                    result.hands[selected].world;

                for (size_t i = 0; i < 21; ++i)
                {
                    coordinates[i * 3 + 0] = landmarks[i].x;
                    coordinates[i * 3 + 1] = landmarks[i].y;
                    coordinates[i * 3 + 2] = landmarks[i].z;
                }

                world = coordinates;
            }

            try
            {
                sock->send(encode_packet(
                    seq,
                    stamp,
                    right,
                    world,
                    cal,
                    false));
            }
            catch (const std::exception& e)
            {
                message = std::string("Send failed: ") + e.what();
            }

            cal = CAL_NONE;

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

            if (alive && status->message != message)
            {
                message = status->message;
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
                       << (world.has_value()
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
                result.hands,
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
                    cal = CAL_OPEN;
                    break;

                case 'f':
                    cal = CAL_FIST;
                    break;

                case 'r':
                    cal = CAL_RESET;
                    break;

                default:
                    cal = CAL_NONE;
                    break;
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "hand_cam error: " << e.what() << "\n";
        exit_code = 2;
    }

    // Match Python shutdown behavior: send three quit packets over UDP.
    for (int i = 0; i < 3; ++i)
    {
        try
        {
            sock->send(encode_packet(
                seq++,
                monotonic_seconds(),
                right,
                std::nullopt,
                CAL_NONE,
                true));
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
