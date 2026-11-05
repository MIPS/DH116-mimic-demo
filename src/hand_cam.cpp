// hand_cam.cpp - camera side of the DH116 mimic demo.
//
// Per camera frame: palm detector -> hand box -> landmark net (both ONNX Runtime) -> 21 hand landmarks in
// metres, sent over UDP to mimic_ctrl (mimic_msg.h). mimic_ctrl replies with the joint targets and the
// measured positions, which the preview window shows. The palm detector only runs every --det-every frames,
// or when no hand is tracked; in between, the hand box comes from the previous frame's landmarks.
//
// Builds with HAND_CAM_HEADLESS use an OpenCV without highgui (no preview window) and always run as --no-preview.
// --web PORT shows the same preview in a web browser (web_view.h), with or without the window.
// --snapshot (with --web) is not real time: the browser shows the live camera, and each press of its Capture button
// runs the nets once on that frame and sends one command to mimic_ctrl. With --camera browser the camera is the
// browser's own (a laptop's webcam, say): each press uploads one picture, and this machine needs no camera.

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include "mimic_msg.h"
#include "web_view.h"

constexpr double ROI_SCALE = 2.9;     // hand box side = 2.9 x palm box side,
constexpr double ROI_SHIFT = 0.5;     // moved half a palm towards the fingers
constexpr int MAX_HANDS = 2;          // palm detections kept per frame
constexpr float MIN_SCORE = 0.5f;     // palm detection score and hand presence needed
constexpr float PALM_NMS_IOU = 0.3f;  // palm boxes overlapping more than this are one hand (uint8 palm model)
constexpr double REPLY_TIMEOUT = 0.5; // seconds without a reply from mimic_ctrl before "No reply"
constexpr int POS_MAX = 10000;        // raw joint position of a closed joint (see dh116.h)
constexpr int PROFILE_EVERY = 30;     // frames per --profile line
constexpr const char* WINDOW_NAME = "DH116 Mimic";
constexpr const char* HELP = "O: open hand   F: fist   R: reset   Q/Esc: quit";

// --snapshot: after a capture, the Capture button unlocks when the hand has reached its target plus SNAPSHOT_SETTLE.
constexpr double SNAPSHOT_ACK_TIMEOUT = 1.0;   // seconds for mimic_ctrl to confirm that it got the command
constexpr double SNAPSHOT_POLL = 0.1;          // seconds between status requests while the hand moves
constexpr double SNAPSHOT_REACH_TIMEOUT = 6.0; // stop waiting for the hand to arrive after this (dh116.c allows 2 s to open)
constexpr double SNAPSHOT_SETTLE = 2.0;        // safety pause between the hand arriving and the next capture
constexpr int SNAPSHOT_REACH_TOLERANCE = 150;  // raw position error that counts as arrived (OPEN_TOLERANCE in dh116.c)
constexpr int SNAPSHOT_REACH_POLLS = 2;        // status replies in a row within the tolerance
constexpr int SNAPSHOT_BROWSER_POLL_MS = 10;   // --camera browser: pause between looks for a press and for replies

const char* const JOINT_NAMES[6] = {"Thumb sweep", "Thumb bend", "Index", "Middle", "Ring", "Little"};

// Landmark pairs drawn as the hand skeleton.
const std::pair<int, int> BONES[] = {{0, 1},  {1, 2},   {2, 3},   {3, 4},   {0, 5},   {5, 6},   {6, 7},
                                     {7, 8},  {5, 9},   {9, 10},  {10, 11}, {11, 12}, {9, 13},  {13, 14},
                                     {14, 15}, {15, 16}, {13, 17}, {17, 18}, {18, 19}, {19, 20}, {0, 17}};

// The preview window; does nothing in headless builds (which never show it).
namespace gui
{
#ifdef HAND_CAM_HEADLESS
inline void show(const cv::Mat&) {}
inline int wait_key() { return -1; }
inline bool visible() { return true; }
#else
inline void show(const cv::Mat& image) { cv::imshow(WINDOW_NAME, image); }
inline int wait_key() { return cv::waitKey(1) & 0xFF; }
inline bool visible() { return cv::getWindowProperty(WINDOW_NAME, cv::WND_PROP_VISIBLE) >= 1; }
#endif
} // namespace gui

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// ---------------------------------------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------------------------------------

struct Options
{
    std::string camera = "0"; // camera index, video file, or "browser" (--snapshot: the pictures come from the web page)
    int width = 320, height = 240;
    std::string hand = "left";
    int port = MSG_PORT;
    std::string model_dir = "models";
    std::string palm_model, landmark_model; // explicit files; empty = the default name inside model_dir
    int fps = 30;                           // frame rate requested from a camera
    bool flip_input = false, no_preview = false, profile = false, bench = false;
    int frames = 0, threads = 8, det_every = 30, bench_iters = 100;
    double freeze_crop = 0.03; // keep the hand crop while its box moves less than this fraction of its size
    int web_port = 0; // --web: TCP port of the browser preview; 0 = off
    bool snapshot = false; // --snapshot: one capture and one command per button press in the browser

    bool browser_camera() const { return camera == "browser"; }

    std::string palm_path() const { return palm_model.empty() ? model_dir + "/palm.onnx" : palm_model; }
    std::string landmark_path() const
    {
        return landmark_model.empty() ? model_dir + "/landmark.onnx" : landmark_model;
    }
};

void print_usage(const char* argv0)
{
    std::cout << "Usage: " << argv0 << " [options]\n\n"
              << "  --camera SOURCE     camera index, video file, or browser (default 0). browser: the camera of the\n"
              << "                      computer whose browser opens the --web page; needs --snapshot (see --snapshot)\n"
              << "  --width N           camera width (default 320); the size asked of the browser's camera too\n"
              << "  --height N          camera height (default 240)\n"
              << "  --fps N             frame rate requested from the camera (default 30; the driver may ignore it)\n"
              << "  --hand left|right   hand to track (default left)\n"
              << "  --port N            mimic_ctrl UDP port (default " << MSG_PORT << ")\n"
              << "  --model-dir DIR     holds palm.onnx and landmark.onnx (default models)\n"
              << "  --palm-model FILE   palm detector, instead of DIR/palm.onnx\n"
              << "  --landmark-model FILE  landmark net, instead of DIR/landmark.onnx\n"
              << "                      (the input size of both is read from the model; it must be square)\n"
              << "  --flip-input        mirror the camera image first (for cameras that deliver it mirrored)\n"
              << "  --no-preview        no preview window\n"
              << "  --web PORT          also show the preview in a web browser at http://<this machine>:PORT/\n"
              << "                      (open to the whole network, no password)\n"
              << "  --snapshot          not real time (needs --web): the page shows the live camera and a Capture button.\n"
              << "                      Each press runs the models once on the current frame and sends one command to the\n"
              << "                      hand; the button unlocks once the hand has arrived, plus " << SNAPSHOT_SETTLE << " s.\n"
              << "                      No O/F/R keys: mimic_ctrl keeps its saved or default calibration.\n"
              << "                      --frames, --det-every, --freeze-crop and --profile do not apply\n"
              << "                      With --camera browser each press sends one picture from the browser's camera. A\n"
              << "                      browser allows its camera only on https:// or http://localhost, so on the computer\n"
              << "                      with the camera run: ssh -L PORT:localhost:PORT USER@<this machine>, and open\n"
              << "                      http://localhost:PORT/\n"
              << "  --frames N          stop after N frames; 0 means unlimited\n"
              << "  --threads N         ONNX Runtime threads per model (default 8)\n"
              << "  --det-every N       run the palm detector every N frames (default 30)\n"
              << "  --freeze-crop F     keep the hand crop while the hand moves less than F x its box size\n"
              << "                      (default 0.03; 0 = re-crop every frame). Steadies the landmarks on a still hand\n"
              << "  --profile           print the average time per stage every " << PROFILE_EVERY << " frames\n"
              << "  --bench             time both models on random input and exit (no camera needed)\n"
              << "  --bench-iters N     timed runs per model for --bench (default 100)\n"
              << "  --help              show this help\n";
}

Options parse_args(int argc, char** argv)
{
    Options o;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        auto value = [&]() -> std::string
        {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + arg);
            return argv[++i];
        };
        if (arg == "--help" || arg == "-h")
        {
            print_usage(argv[0]);
            std::exit(0);
        }
        else if (arg == "--camera") o.camera = value();
        else if (arg == "--width") o.width = std::stoi(value());
        else if (arg == "--height") o.height = std::stoi(value());
        else if (arg == "--hand") o.hand = value();
        else if (arg == "--port") o.port = std::stoi(value());
        else if (arg == "--fps") o.fps = std::stoi(value());
        else if (arg == "--model-dir") o.model_dir = value();
        else if (arg == "--palm-model") o.palm_model = value();
        else if (arg == "--landmark-model") o.landmark_model = value();
        else if (arg == "--flip-input") o.flip_input = true;
        else if (arg == "--no-preview") o.no_preview = true;
        else if (arg == "--web") o.web_port = std::stoi(value());
        else if (arg == "--snapshot") o.snapshot = true;
        else if (arg == "--frames") o.frames = std::stoi(value());
        else if (arg == "--threads") o.threads = std::stoi(value());
        else if (arg == "--det-every") o.det_every = std::stoi(value());
        else if (arg == "--freeze-crop") o.freeze_crop = std::stod(value());
        else if (arg == "--profile") o.profile = true;
        else if (arg == "--bench") o.bench = true;
        else if (arg == "--bench-iters") o.bench_iters = std::stoi(value());
        else throw std::runtime_error("unknown argument: " + arg);
    }
    if (o.hand != "left" && o.hand != "right") throw std::runtime_error("--hand must be left or right");
    if (o.port < 1 || o.port > 65535) throw std::runtime_error("--port must be 1..65535");
    if (o.web_port < 0 || o.web_port > 65535) throw std::runtime_error("--web must be 1..65535");
    if (o.snapshot && !o.web_port) throw std::runtime_error("--snapshot needs --web PORT (the Capture button is on the web page)");
    if (o.browser_camera() && !o.snapshot && !o.bench)
        throw std::runtime_error("--camera browser needs --snapshot and --web PORT (the pictures come from the web page)");
    if (o.width <= 0 || o.height <= 0 || o.fps <= 0)
        throw std::runtime_error("--width, --height and --fps must be positive");
    if (o.frames < 0) throw std::runtime_error("--frames must not be negative");
    if (o.threads <= 0 || o.det_every <= 0 || o.bench_iters <= 0)
        throw std::runtime_error("--threads, --det-every and --bench-iters must be positive");
    if (o.freeze_crop < 0 || o.freeze_crop > 0.5) throw std::runtime_error("--freeze-crop must be 0..0.5");
    return o;
}

bool is_number(const std::string& s)
{
    return !s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c) { return c >= '0' && c <= '9'; });
}

// ---------------------------------------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------------------------------------

// A number opens that V4L2 camera (MJPG, --fps, --width x --height); anything else is opened as a video file.
cv::VideoCapture open_camera(const Options& o)
{
    const bool is_camera = is_number(o.camera);
    cv::VideoCapture cap;
    if (is_camera) cap.open(std::stoi(o.camera), cv::CAP_V4L2);
    else cap.open(o.camera);
    if (!cap.isOpened()) throw std::runtime_error("could not open camera or video: " + o.camera);
    if (is_camera)
    {
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        cap.set(cv::CAP_PROP_FRAME_WIDTH, o.width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, o.height);
        cap.set(cv::CAP_PROP_FPS, o.fps);
    }
    return cap;
}

// Reads the camera on its own thread and keeps only the newest frame, so capture overlaps inference and the
// robot never lags behind by queued frames.
class Grabber
{
public:
    struct Frame
    {
        cv::Mat image;
        double stamp = 0; // capture time, now_s()
        uint64_t count = 0; // frame number, 1 = first frame
    };

    explicit Grabber(cv::VideoCapture& cap) : cap_(cap), thread_(&Grabber::run, this) {}
    ~Grabber()
    {
        running_ = false;
        thread_.join();
    }

    // Waits up to 1 s for a frame newer than frame number `last`. On a timeout the result's count is still `last`.
    Frame next(uint64_t last)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        new_frame_.wait_for(lock, std::chrono::seconds(1), [&] { return newest_.count != last || failures_ > 30; });
        if (failures_ > 30) throw std::runtime_error("the camera stopped delivering frames");
        return newest_;
    }

private:
    void run()
    {
        while (running_)
        {
            cv::Mat image;
            const bool ok = cap_.read(image);
            const double stamp = now_s();
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (ok)
                {
                    newest_ = {image, stamp, newest_.count + 1};
                    failures_ = 0;
                }
                else ++failures_;
            }
            new_frame_.notify_all();
            if (!ok) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    cv::VideoCapture& cap_;
    std::atomic<bool> running_{true};
    std::mutex mutex_;
    std::condition_variable new_frame_;
    Frame newest_;
    int failures_ = 0; // failed reads in a row
    std::thread thread_; // last: starts after everything above is initialised
};

// ---------------------------------------------------------------------------------------------------------
// Hand landmarks
// ---------------------------------------------------------------------------------------------------------

// A square hand box in image pixels, turned by `rotation` (radians) so that the fingers point up inside it.
struct ROI
{
    double cx = 0, cy = 0, size = 0, rotation = 0;
};

struct Hand
{
    std::array<cv::Point2f, 21> points; // landmarks in image pixels
    std::array<float, 63> world;        // landmarks in metres, x y z per point
    bool right = false;                 // the net sees a right hand
    float side_score = 0;               // how sure it is about left/right, 0.5..1
};

// The float data of one model output, checking that it has at least min_count values.
const float* floats(const Ort::Value& output, size_t min_count)
{
    const auto info = output.GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || info.GetElementCount() < min_count)
        throw std::runtime_error("unexpected model output");
    return output.GetTensorData<float>();
}

// Quantisation of a uint8 model output: value = (byte - zero) * scale. The Qualcomm w8a8 models keep these
// numbers in their metadata.json, not in the .onnx file.
struct Quant
{
    float scale;
    int zero;
};
constexpr Quant PALM_BOX_Q = {1.6417323f, 50}; // box_coords, in pixels of the palm input
constexpr Quant PALM_SCORE_Q = {1.0f / 256, 0};
constexpr Quant LM_XYZ_Q = {1.1388181f, 48};   // landmarks, in pixels of the crop
constexpr Quant LM_SCORE_Q = {1.0f / 256, 0};  // hand presence and left/right
constexpr Quant LM_WORLD_Q = {0.0009378f, 122}; // world landmarks, in metres

// Like floats(), but a uint8 output is dequantised with q into buf first.
const float* floats(const Ort::Value& output, size_t min_count, const Quant& q, std::vector<float>& buf)
{
    const auto info = output.GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) return floats(output, min_count);
    if (info.GetElementCount() < min_count) throw std::runtime_error("unexpected model output");
    const uint8_t* bytes = output.GetTensorData<uint8_t>();
    buf.resize(info.GetElementCount());
    for (size_t i = 0; i < buf.size(); ++i) buf[i] = (static_cast<int>(bytes[i]) - q.zero) * q.scale;
    return buf.data();
}

// One ONNX model. Its input layout (NCHW or NHWC) is taken from the model's input shape.
class Model
{
public:
    Model(Ort::Env& env, const std::string& path, int threads)
    {
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(threads);
        options.SetInterOpNumThreads(1);
        session_ = Ort::Session(env, path.c_str(), options);

        Ort::AllocatorWithDefaultOptions alloc;
        input_name_ = session_.GetInputNameAllocated(0, alloc).get();
        for (size_t i = 0; i < session_.GetOutputCount(); ++i)
            output_names_.emplace_back(session_.GetOutputNameAllocated(i, alloc).get());
        for (const auto& name : output_names_) output_ptrs_.push_back(name.c_str());

        input_shape = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        nchw_ = input_shape.size() == 4 && input_shape[1] == 3;
        uint8_ = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType() ==
                 ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    }

    // Runs the model on an RGB uint8 image of its input size. A float model gets pixel values scaled to 0..1;
    // a quantised model with a uint8 input gets the bytes as they are (the quantised models take 0..1 with scale 1/255).
    std::vector<Ort::Value> run(const cv::Mat& rgb)
    {
        const int rows = rgb.rows, cols = rgb.cols;
        const size_t plane = static_cast<size_t>(rows) * cols;
        std::vector<int64_t> shape = {1, rows, cols, 3};
        if (nchw_) shape = {1, 3, rows, cols};
        if (uint8_)
        {
            input8_.resize(plane * 3);
            if (nchw_)
            {
                cv::Mat planes[3] = {cv::Mat(rows, cols, CV_8U, input8_.data()),
                                     cv::Mat(rows, cols, CV_8U, input8_.data() + plane),
                                     cv::Mat(rows, cols, CV_8U, input8_.data() + 2 * plane)};
                cv::split(rgb, planes);
            }
            else
            {
                cv::Mat dst(rows, cols, CV_8UC3, input8_.data());
                rgb.copyTo(dst);
            }
            Ort::Value input = Ort::Value::CreateTensor<uint8_t>(memory_, input8_.data(), input8_.size(),
                                                                  shape.data(), shape.size());
            return run(input);
        }
        input_.resize(plane * 3);
        if (nchw_) // one float plane per colour
        {
            rgb.convertTo(scratch_, CV_32FC3, 1.0 / 255.0);
            cv::Mat planes[3] = {cv::Mat(rows, cols, CV_32F, input_.data()),
                                 cv::Mat(rows, cols, CV_32F, input_.data() + plane),
                                 cv::Mat(rows, cols, CV_32F, input_.data() + 2 * plane)};
            cv::split(scratch_, planes);
        }
        else // interleaved, like the image
        {
            cv::Mat dst(rows, cols, CV_32FC3, input_.data());
            rgb.convertTo(dst, CV_32FC3, 1.0 / 255.0);
        }
        Ort::Value input =
            Ort::Value::CreateTensor<float>(memory_, input_.data(), input_.size(), shape.data(), shape.size());
        return run(input);
    }

    // Runs the model on a ready input tensor (used by --bench).
    std::vector<Ort::Value> run(Ort::Value& input)
    {
        const char* name = input_name_.c_str();
        return session_.Run(Ort::RunOptions{nullptr}, &name, &input, 1, output_ptrs_.data(), output_ptrs_.size());
    }

    std::vector<int64_t> input_shape; // as declared by the model; -1 for a free dimension
    bool uint8_input() const { return uint8_; }

    // Side of the square input image, from the model's shape (NCHW or NHWC).
    int size() const
    {
        const int64_t h = nchw_ ? input_shape.at(2) : input_shape.at(1), w = nchw_ ? input_shape.at(3) : input_shape.at(2);
        if (input_shape.size() != 4 || h <= 0 || h != w)
            throw std::runtime_error("the model input must be a fixed, square 4-D image");
        return static_cast<int>(h);
    }

private:
    Ort::Session session_{nullptr};
    std::string input_name_;
    std::vector<std::string> output_names_;
    std::vector<const char*> output_ptrs_; // point into output_names_
    bool nchw_ = false;
    bool uint8_ = false; // the model's input is uint8 (quantised) instead of float
    Ort::MemoryInfo memory_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<float> input_;    // reused input buffer (float models)
    std::vector<uint8_t> input8_; // reused input buffer (uint8 models)
    cv::Mat scratch_;          // reused float image for NCHW
};

// Finds hands and their landmarks. The hand boxes for the next frame come from this frame's landmarks, so the
// palm detector only runs every det_every frames, or when no hand is tracked.
class HandLandmarker
{
public:
    HandLandmarker(const std::string& palm_path, const std::string& landmark_path, int threads, int det_every,
                   double freeze = 0)
        : env_(ORT_LOGGING_LEVEL_WARNING, "DH116Hand"), palm_(env_, palm_path, threads),
          landmark_(env_, landmark_path, threads), det_every_(det_every), freeze_(freeze), palm_size_(palm_.size()),
          lm_size_(landmark_.size())
    {
    }

    double palm_ms = 0, landmark_ms = 0; // time spent per stage, for --profile; the caller resets them

    // Hands in an RGB image. Once a hand of the wanted side is found, only that one is tracked further.
    std::vector<Hand> find(const cv::Mat& rgb, bool want_right)
    {
        ++frames_;
        if (boxes_.empty() || frames_ % det_every_ == 0)
        {
            const double t0 = now_s();
            std::vector<ROI> found = palm_boxes(rgb);
            if (freeze_ > 0) // a re-detection of a hand already tracked keeps the tracked box
                for (ROI& b : found)
                    for (const ROI& t : boxes_)
                        if (std::hypot(b.cx - t.cx, b.cy - t.cy) < 0.25 * t.size)
                        {
                            b = t;
                            break;
                        }
            boxes_ = found;
            palm_ms += (now_s() - t0) * 1e3;
        }

        std::vector<Hand> hands;
        std::vector<ROI> next_boxes, wanted_boxes;
        for (const ROI& box : boxes_)
        {
            const double t0 = now_s();
            cv::Matx23d to_crop;
            const auto out = landmark_.run(crop(rgb, box, to_crop));
            landmark_ms += (now_s() - t0) * 1e3;

            if (out.size() < 4) throw std::runtime_error("the landmark model must have 4 outputs");
            std::vector<float> xyz_buf, presence_buf, side_buf, world_buf; // used by uint8 models only
            const float* crop_xyz = floats(out[0], 63, LM_XYZ_Q, xyz_buf); // x y z per landmark, in crop pixels
            const float presence = floats(out[1], 1, LM_SCORE_Q, presence_buf)[0];
            const float side = floats(out[2], 1, LM_SCORE_Q, side_buf)[0]; // > 0.5: right hand
            const float* world = floats(out[3], 63, LM_WORLD_Q, world_buf);
            if (presence < MIN_SCORE) continue;

            Hand hand;
            hand.right = side > 0.5f;
            hand.side_score = hand.right ? side : 1.0f - side;
            cv::Matx23d to_image;
            cv::invertAffineTransform(to_crop, to_image);
            for (int i = 0; i < 21; ++i)
            {
                const cv::Vec2d p = to_image * cv::Vec3d(crop_xyz[3 * i], crop_xyz[3 * i + 1], 1.0);
                hand.points[i] = cv::Point2f(static_cast<float>(p[0]), static_cast<float>(p[1]));
            }
            std::copy_n(world, 63, hand.world.begin());

            hands.push_back(hand);
            const ROI next = box_from_landmarks(hand.points);
            next_boxes.push_back(nearly_same(next, box) ? box : next);
            if (hand.right == want_right) wanted_boxes.push_back(next_boxes.back());
        }
        boxes_ = wanted_boxes.empty() ? next_boxes : wanted_boxes;
        return hands;
    }

private:
    // Palm detector on a letterboxed copy of the image -> up to MAX_HANDS hand boxes, best first.
    std::vector<ROI> palm_boxes(const cv::Mat& rgb)
    {
        // Letterbox: scale the longer side to palm_size_ and pad the shorter one with black.
        const int PALM_SIZE = palm_size_;
        const double scale = static_cast<double>(PALM_SIZE) / std::max(rgb.rows, rgb.cols);
        const int w = static_cast<int>(rgb.cols * scale), h = static_cast<int>(rgb.rows * scale);
        const int pad_x = (PALM_SIZE - w) / 2, pad_y = (PALM_SIZE - h) / 2;
        cv::Mat input = cv::Mat::zeros(PALM_SIZE, PALM_SIZE, CV_8UC3);
        cv::resize(rgb, input(cv::Rect(pad_x, pad_y, w, h)), cv::Size(w, h));

        // One row per detection: score, box centre x y, box size, wrist x y, middle finger base x y,
        // all relative to the palm input (0..1).
        const auto out = palm_.run(input);
        std::vector<std::array<float, 8>> found;
        if (palm_.uint8_input())
            found = palm_rows_uint8(out);
        else
        {
            const auto info = out.at(0).GetTensorTypeAndShapeInfo();
            if (info.GetShape().empty() || info.GetShape().back() != 8)
                throw std::runtime_error("the palm model output must have 8 values per detection");
            const float* rows = floats(out[0], 0);
            for (size_t i = 0; i < info.GetElementCount() / 8; ++i)
                if (rows[8 * i] >= MIN_SCORE) found.push_back({rows[8 * i], rows[8 * i + 1], rows[8 * i + 2],
                                                               rows[8 * i + 3], rows[8 * i + 4], rows[8 * i + 5],
                                                               rows[8 * i + 6], rows[8 * i + 7]});
        }
        std::sort(found.begin(), found.end(), [](const auto& a, const auto& b) { return a[0] > b[0]; });
        if (found.size() > MAX_HANDS) found.resize(MAX_HANDS);

        // Undo the letterbox: palm input (0..1) -> image pixels.
        auto px = [&](double v, int pad) { return (v * PALM_SIZE - pad) / scale; };
        std::vector<ROI> boxes;
        for (const auto& d : found)
        {
            const double cx = px(d[1], pad_x), cy = px(d[2], pad_y);
            const double size = static_cast<double>(d[3]) * PALM_SIZE / scale;
            const double wrist_x = px(d[4], pad_x), wrist_y = px(d[5], pad_y);
            const double finger_x = px(d[6], pad_x), finger_y = px(d[7], pad_y);
            // Turn the wrist -> middle finger direction to point up (image y grows downwards).
            const double rotation = 0.5 * M_PI - std::atan2(-(finger_y - wrist_y), finger_x - wrist_x);
            boxes.push_back({cx + ROI_SHIFT * size * std::sin(rotation), cy - ROI_SHIFT * size * std::cos(rotation),
                             ROI_SCALE * size, rotation});
        }
        return boxes;
    }

    // The uint8 palm model returns the raw detections of all anchors: box_coords [1,N,18] (box x0 y0 x1 y1, then
    // 7 keypoints x y, in pixels of the palm input) and box_scores [1,N]. Boxes that overlap are one hand, and
    // are averaged by score. Returns rows in the format of the float model, best first.
    std::vector<std::array<float, 8>> palm_rows_uint8(const std::vector<Ort::Value>& out) const
    {
        constexpr size_t VALUES = 18;
        const auto shape = out.at(0).GetTensorTypeAndShapeInfo().GetShape();
        if (out.size() < 2 || shape.empty() || shape.back() != VALUES)
            throw std::runtime_error("the uint8 palm model needs outputs box_coords [1,N,18] and box_scores [1,N]");
        const size_t n = out[0].GetTensorTypeAndShapeInfo().GetElementCount() / VALUES;
        std::vector<float> coord_buf, score_buf;
        const float* coords = floats(out[0], n * VALUES, PALM_BOX_Q, coord_buf);
        const float* scores = floats(out[1], n, PALM_SCORE_Q, score_buf);

        std::vector<size_t> order;
        for (size_t i = 0; i < n; ++i)
            if (scores[i] >= MIN_SCORE) order.push_back(i);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return scores[a] > scores[b]; });

        auto iou = [&](const float* a, const float* b) {
            const float w = std::min(a[2], b[2]) - std::max(a[0], b[0]), h = std::min(a[3], b[3]) - std::max(a[1], b[1]);
            if (w <= 0 || h <= 0) return 0.0f;
            const float inter = w * h, areas = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]);
            return inter / (areas - inter);
        };
        std::vector<std::array<float, 8>> rows;
        std::vector<bool> used(order.size(), false);
        const float S = static_cast<float>(palm_size_);
        for (size_t i = 0; i < order.size(); ++i)
        {
            if (used[i]) continue;
            std::array<float, VALUES> mean{};
            float weight = 0;
            for (size_t j = i; j < order.size(); ++j)
            {
                if (used[j] || (j != i && iou(coords + order[i] * VALUES, coords + order[j] * VALUES) < PALM_NMS_IOU))
                    continue;
                used[j] = true;
                weight += scores[order[j]];
                for (size_t k = 0; k < VALUES; ++k) mean[k] += scores[order[j]] * coords[order[j] * VALUES + k];
            }
            for (float& v : mean) v /= weight;
            rows.push_back({scores[order[i]], (mean[0] + mean[2]) / 2 / S, (mean[1] + mean[3]) / 2 / S,
                            std::max(mean[2] - mean[0], mean[3] - mean[1]) / S, mean[4] / S, mean[5] / S,
                            mean[8] / S, mean[9] / S});
        }
        return rows;
    }

    // lm_size_ x lm_size_ crop of a hand box, fingers up. to_crop maps image pixels to crop pixels.
    cv::Mat crop(const cv::Mat& rgb, const ROI& box, cv::Matx23d& to_crop) const
    {
        const int LM_SIZE = lm_size_;
        const double scale = LM_SIZE / box.size, c = std::cos(-box.rotation), s = std::sin(-box.rotation);
        to_crop = cv::Matx23d(scale * c, -scale * s, LM_SIZE / 2.0 - scale * (c * box.cx - s * box.cy), //
                              scale * s, scale * c, LM_SIZE / 2.0 - scale * (s * box.cx + c * box.cy));
        cv::Mat out;
        cv::warpAffine(rgb, out, to_crop, cv::Size(LM_SIZE, LM_SIZE));
        return out;
    }

    // Hand box around the landmarks: turned by the wrist (0) -> middle finger base (9) direction, centred on
    // their bounding box, 1.2 x its diagonal in size.
    static ROI box_from_landmarks(const std::array<cv::Point2f, 21>& p)
    {
        const double dx = p[9].x - p[0].x, dy = p[9].y - p[0].y;
        double min_x = p[0].x, max_x = p[0].x, min_y = p[0].y, max_y = p[0].y;
        for (const auto& q : p)
        {
            min_x = std::min<double>(min_x, q.x);
            max_x = std::max<double>(max_x, q.x);
            min_y = std::min<double>(min_y, q.y);
            max_y = std::max<double>(max_y, q.y);
        }
        return {(min_x + max_x) * 0.5, (min_y + max_y) * 0.5, std::hypot(max_x - min_x, max_y - min_y) * 1.2,
                0.5 * M_PI - std::atan2(-dy, dx)};
    }

    // Frozen crop: while the hand box moves less than freeze_ of its size (and turns and grows less than twice
    // that), the previous box is kept, so a still hand is cropped identically every frame and the landmark noise
    // is not fed back into the next crop.
    bool nearly_same(const ROI& a, const ROI& b) const
    {
        if (freeze_ <= 0) return false;
        const double turn = std::remainder(a.rotation - b.rotation, 2 * M_PI);
        return std::hypot(a.cx - b.cx, a.cy - b.cy) < freeze_ * b.size &&
               std::abs(a.size / b.size - 1.0) < 2 * freeze_ && std::abs(turn) < 2 * freeze_;
    }

    Ort::Env env_;
    Model palm_, landmark_;
    int det_every_;
    double freeze_;
    int palm_size_, lm_size_; // input side of each model, pixels
    uint64_t frames_ = 0;
    std::vector<ROI> boxes_; // hand boxes to look in on the next frame
};

// ---------------------------------------------------------------------------------------------------------
// UDP link to mimic_ctrl
// ---------------------------------------------------------------------------------------------------------

// Non-blocking UDP socket to mimic_ctrl on 127.0.0.1: sending never waits, replies are polled.
class Link
{
public:
    explicit Link(int port)
    {
        fd_ = ::socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK, 0);
        if (fd_ < 0) throw std::runtime_error("could not create the UDP socket");
        to_.sin_family = AF_INET;
        to_.sin_port = htons(static_cast<uint16_t>(port));
        to_.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    }
    ~Link() { ::close(fd_); }
    Link(const Link&) = delete;
    Link& operator=(const Link&) = delete;

    bool send(const landmarks_msg& m)
    {
        return sendto(fd_, &m, sizeof(m), 0, reinterpret_cast<const sockaddr*>(&to_), sizeof(to_)) ==
               static_cast<ssize_t>(sizeof(m));
    }

    // The newest valid reply received since the last call, if any.
    std::optional<status_msg> newest_reply()
    {
        std::optional<status_msg> newest;
        status_msg s;
        ssize_t n;
        while ((n = recv(fd_, &s, sizeof(s), MSG_TRUNC)) >= 0)
        {
            if (n != static_cast<ssize_t>(sizeof(s)) || s.version != MSG_VERSION) continue;
            s.text[sizeof(s.text) - 1] = '\0';
            newest = s;
            ++replies_;
        }
        return newest;
    }

    // Valid replies received so far, also those newest_reply() skipped because a newer one followed. mimic_ctrl
    // sends one per message, so --snapshot counts them to know which message a reply answers.
    uint64_t replies() const { return replies_; }

private:
    int fd_ = -1;
    sockaddr_in to_{};
    uint64_t replies_ = 0;
};

// ---------------------------------------------------------------------------------------------------------
// Preview
// ---------------------------------------------------------------------------------------------------------

// Draws the hand skeletons (the tracked hand in colour), mirrors the image so the operator sees a mirror view,
// then adds the text lines at the top and one bar per joint: target filled orange, measured as a green tick.
// bars = false leaves the bar box out (--snapshot, while there is nothing to show in it).
void draw_preview(cv::Mat& image, const std::vector<Hand>& hands, int selected, const status_msg* status,
                  const std::vector<std::string>& lines, bool bars = true)
{
    for (size_t i = 0; i < hands.size(); ++i)
    {
        const bool tracked = static_cast<int>(i) == selected;
        const cv::Scalar color = tracked ? cv::Scalar(80, 220, 80) : cv::Scalar(150, 150, 150);
        const auto& p = hands[i].points;
        for (const auto& bone : BONES) cv::line(image, p[bone.first], p[bone.second], color, 2);
        for (const auto& point : p) cv::circle(image, point, 3, tracked ? cv::Scalar(40, 40, 230) : color, -1);
    }
    cv::flip(image, image, 1);

    const cv::Scalar black(0, 0, 0), white(255, 255, 255);
    const int h = image.rows;
    cv::rectangle(image, cv::Point(0, 0), cv::Point(image.cols, 14 + 20 * static_cast<int>(lines.size())), black, -1);
    if (bars) cv::rectangle(image, cv::Point(0, h - 30 - 20 * 6), cv::Point(272, h), black, -1);
    for (size_t r = 0; r < lines.size(); ++r)
        cv::putText(image, lines[r], cv::Point(10, 22 + 20 * static_cast<int>(r)), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    white, 1);
    if (!bars) return;

    auto bar_x = [](int pos) { return 110 + static_cast<int>(150.0 * std::clamp(pos, 0, POS_MAX) / POS_MAX); };
    for (int j = 0; j < 6; ++j)
    {
        const int y = h - 12 - 20 * (5 - j);
        cv::putText(image, JOINT_NAMES[j], cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.45, white, 1);
        cv::rectangle(image, cv::Point(110, y - 11), cv::Point(260, y + 1), cv::Scalar(60, 60, 60), -1);
        if (!status) continue;
        cv::rectangle(image, cv::Point(110, y - 11), cv::Point(bar_x(status->target[j]), y + 1),
                      cv::Scalar(80, 200, 255), -1);
        if (status->got_feedback)
        {
            const int x = bar_x(status->measured[j]);
            cv::line(image, cv::Point(x, y - 14), cv::Point(x, y + 4), cv::Scalar(80, 255, 80), 2);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// --profile and --bench
// ---------------------------------------------------------------------------------------------------------

// Average milliseconds per stage, printed every PROFILE_EVERY frames.
struct Profile
{
    double wait_ms = 0, find_ms = 0, show_ms = 0;
    int n = 0;

    void add(double wait_s, double find_s, double show_s, HandLandmarker& lm)
    {
        wait_ms += wait_s * 1e3;
        find_ms += find_s * 1e3;
        show_ms += show_s * 1e3;
        if (++n < PROFILE_EVERY) return;
        std::printf("profile/frame ms: wait-camera %.1f | detect %.1f (palm %.1f, landmark %.1f) | draw+show %.1f\n",
                    wait_ms / n, find_ms / n, lm.palm_ms / n, lm.landmark_ms / n, show_ms / n);
        std::fflush(stdout);
        wait_ms = find_ms = show_ms = lm.palm_ms = lm.landmark_ms = 0;
        n = 0;
    }
};

// Times each model alone on random input, like scripts/bench_models.py but with the ONNX Runtime linked here.
int run_bench(const Options& opt)
{
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "DH116Bench");
    const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    const std::string files[2] = {opt.palm_path(), opt.landmark_path()};
    double median_ms[2] = {0, 0};

    std::cout << "threads " << opt.threads << ", " << opt.bench_iters << " timed runs, models " << files[0] << " " << files[1]
              << "\n\n   model    |   median     mean      p95      min  (ms)\n";
    for (int m = 0; m < 2; ++m)
    {
        Model model(env, files[m], opt.threads);
        std::vector<int64_t> shape = model.input_shape;
        size_t count = 1;
        for (auto& d : shape) count *= static_cast<size_t>(d = std::max<int64_t>(d, 1)); // free dimension -> 1
        std::vector<float> data(model.uint8_input() ? 0 : count);
        std::vector<uint8_t> data8(model.uint8_input() ? count : 0);
        for (auto& v : data) v = dist(rng);
        for (auto& v : data8) v = static_cast<uint8_t>(rng() & 0xFF);
        Ort::Value input = model.uint8_input()
                               ? Ort::Value::CreateTensor<uint8_t>(memory, data8.data(), data8.size(), shape.data(),
                                                                   shape.size())
                               : Ort::Value::CreateTensor<float>(memory, data.data(), data.size(), shape.data(),
                                                                 shape.size());

        std::vector<double> ms;
        for (int i = 0; i < 10 + opt.bench_iters; ++i) // the first 10 runs are warm-up
        {
            const double t0 = now_s();
            model.run(input);
            if (i >= 10) ms.push_back((now_s() - t0) * 1e3);
        }
        std::sort(ms.begin(), ms.end());
        double sum = 0;
        for (double v : ms) sum += v;
        median_ms[m] = ms[ms.size() / 2];
        std::printf("%-10s  | %8.2f %8.2f %8.2f %8.2f\n", m == 0 ? "palm" : "landmark", median_ms[m], sum / ms.size(),
                    ms[std::min(ms.size() - 1, static_cast<size_t>(ms.size() * 0.95))], ms[0]);
    }

    const double steady = median_ms[1], average = steady + median_ms[0] / opt.det_every;
    std::printf("\nper frame (palm every %d): steady %.2f ms, average %.2f ms, spike %.2f ms, %.1f fps\n",
                opt.det_every, steady, average, steady + median_ms[0], 1000.0 / average);
    return 0;
}

// ---------------------------------------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------------------------------------

// The hand to follow: the one of the wanted side the net is most sure about, or -1.
int select_hand(const std::vector<Hand>& hands, bool right)
{
    int selected = -1;
    for (size_t i = 0; i < hands.size(); ++i)
        if (hands[i].right == right && (selected < 0 || hands[i].side_score >= hands[selected].side_score))
            selected = static_cast<int>(i);
    return selected;
}

// The message for one frame; a null hand (none of the wanted side found) makes it a "no hand" message.
landmarks_msg make_message(bool right, double stamp, const Hand* hand)
{
    landmarks_msg msg{};
    msg.version = MSG_VERSION;
    msg.flags = right ? MSG_RIGHT_HAND : 0;
    msg.time_s = stamp;
    if (hand)
    {
        msg.flags |= MSG_HAND_SEEN;
        std::memcpy(msg.world, hand->world.data(), sizeof(msg.world));
    }
    return msg;
}

// Tells mimic_ctrl to open the hand and stop; three times, as UDP may drop one.
void send_quit(Link& link)
{
    landmarks_msg quit{};
    quit.version = MSG_VERSION;
    quit.flags = MSG_QUIT;
    for (int i = 0; i < 3; ++i)
    {
        link.send(quit);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

int run(const Options& opt)
{
    HandLandmarker landmarker(opt.palm_path(), opt.landmark_path(), opt.threads, opt.det_every, opt.freeze_crop);
    cv::VideoCapture cap = open_camera(opt);
    std::unique_ptr<Grabber> grabber; // cameras only; a video file is read frame by frame
    if (is_number(opt.camera)) grabber = std::make_unique<Grabber>(cap);
    Link link(opt.port);
    std::unique_ptr<WebView> web; // --web only
    if (opt.web_port) web = std::make_unique<WebView>(opt.web_port);

    const bool right = opt.hand == "right";
    std::cout << "Tracking your " << opt.hand << " hand -> mimic_ctrl on 127.0.0.1:" << opt.port << ". " << HELP
              << std::endl;
    if (web) std::cout << "Preview for a web browser: http://<this machine>:" << opt.web_port << "/" << std::endl;

    uint64_t last_frame = 0;      // grabber frame number of the last frame used
    int frames = 0;               // frames processed
    int calibrate = MSG_CAL_NONE; // key pressed on the previous frame, sent with the next message
    double fps = 0, previous = now_s();
    std::optional<status_msg> status; // newest reply from mimic_ctrl
    double status_time = 0;
    std::string message; // newest status text (or send error); printed when mimic_ctrl changes it
    Profile profile;

    int exit_code = 0;
    try
    {
        while (!g_stop && (opt.frames == 0 || frames < opt.frames))
        {
            const double t_loop = now_s();
            cv::Mat image;
            double stamp;
            if (grabber)
            {
                const Grabber::Frame frame = grabber->next(last_frame);
                if (frame.count == last_frame) continue; // no new frame within 1 s
                image = frame.image;
                stamp = frame.stamp;
                last_frame = frame.count;
            }
            else
            {
                if (!cap.read(image)) break; // end of the video file
                stamp = now_s();
            }
            const double t_got = now_s();

            if (opt.flip_input) cv::flip(image, image, 1);
            cv::Mat rgb;
            cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
            const std::vector<Hand> hands = landmarker.find(rgb, right);
            const double t_found = now_s();

            // The hand to follow: the one of the wanted side the net is most sure about.
            const int selected = select_hand(hands, right);

            landmarks_msg msg = make_message(right, stamp, selected >= 0 ? &hands[selected] : nullptr);
            msg.calibrate = static_cast<uint8_t>(calibrate);
            if (!link.send(msg)) message = std::string("Send failed: ") + std::strerror(errno);
            calibrate = MSG_CAL_NONE;

            if (auto reply = link.newest_reply())
            {
                status = reply;
                status_time = now_s();
            }
            const double now = now_s();
            ++frames;
            fps = frames > 1 ? 0.9 * fps + 0.1 / std::max(now - previous, 1e-3) : 0.0;
            previous = now;
            const bool alive = status && now - status_time < REPLY_TIMEOUT;
            if (alive && message != status->text)
            {
                message = status->text;
                std::cout << message << std::endl;
            }

            int key = -1;
            const double t_show = now_s();
            const bool draw = !opt.no_preview || web;
            if (draw)
            {
                std::ostringstream top;
                top << (right ? "RIGHT" : "LEFT") << " hand   " << std::fixed << std::setprecision(1) << fps
                    << " FPS   " << (selected >= 0 ? "Hand detected" : "No " + opt.hand + " hand visible");
                std::string state = "No reply from mimic_ctrl on port " + std::to_string(opt.port);
                if (alive) state = status->dry_run ? "DRY RUN (no EtherCAT)" : "hand RUNNING";
                draw_preview(image, hands, selected, alive ? &*status : nullptr, {top.str(), state, message, HELP});
                if (web)
                {
                    web->publish(image);
                    key = web->take_key();
                }
                if (!opt.no_preview)
                {
                    gui::show(image);
                    const int window_key = gui::wait_key();
                    if (key < 0) key = window_key;
                }
            }
            if (opt.profile)
                profile.add(t_got - t_loop, t_found - t_got, draw ? now_s() - t_show : 0, landmarker);

            if (key == 'q' || key == 27 || (!opt.no_preview && !gui::visible())) break;
            if (key == 'o') calibrate = MSG_CAL_OPEN;
            if (key == 'f') calibrate = MSG_CAL_FIST;
            if (key == 'r') calibrate = MSG_CAL_RESET;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "hand_cam error: " << e.what() << "\n";
        exit_code = 2;
    }

    send_quit(link);
    return exit_code;
}

// ---------------------------------------------------------------------------------------------------------
// --snapshot: one capture and one command per button press
// ---------------------------------------------------------------------------------------------------------

// True if every joint is within SNAPSHOT_REACH_TOLERANCE of its target.
bool at_target(const status_msg& s)
{
    for (int j = 0; j < 6; ++j)
        if (std::abs(s.measured[j] - s.target[j]) > SNAPSHOT_REACH_TOLERANCE) return false;
    return true;
}

// The browser shows the live camera. Each press of its Capture button runs the palm detector and the landmark net
// once on the newest frame, sends that one result to mimic_ctrl and shows the landmarks on the captured image.
// The button stays locked until the hand has reached its target (or SNAPSHOT_REACH_TIMEOUT has passed) plus
// SNAPSHOT_SETTLE, so one press moves the hand once. mimic_ctrl is not changed: it answers every message with the
// target it holds and the positions the hand reports, which is what this loop waits on.
int run_snapshot(const Options& opt)
{
    // Palm detector on every capture, no crop freezing: nothing is carried over from the previous capture.
    HandLandmarker landmarker(opt.palm_path(), opt.landmark_path(), opt.threads, 1, 0);
    const bool browser = opt.browser_camera(); // the pictures come with the button presses; nothing to open here
    cv::VideoCapture cap = browser ? cv::VideoCapture() : open_camera(opt);
    std::unique_ptr<Grabber> grabber; // cameras only; a video file is read in this loop and played again at its end
    if (is_number(opt.camera)) grabber = std::make_unique<Grabber>(cap);
    Link link(opt.port);
    WebView web(opt.web_port, true, browser ? opt.width : 0, browser ? opt.height : 0);

    const bool right = opt.hand == "right";
    std::cout << "Snapshot mode: tracking your " << opt.hand << " hand -> mimic_ctrl on 127.0.0.1:" << opt.port << ".\n";
    if (browser)
    {
        const std::string port = std::to_string(opt.web_port);
        std::cout << "The camera is the one of the computer whose browser opens the page. A browser allows its camera only\n"
                  << "on https:// or http://localhost, so on that computer run:\n"
                  << "    ssh -L " << port << ":localhost:" << port << " <user>@<this machine>\n"
                  << "and open http://localhost:" << port << "/ and press Capture." << std::endl;
    }
    else std::cout << "Open http://<this machine>:" << opt.web_port << "/ and press Capture." << std::endl;

    enum class Phase { Live, Ack, Reach, Settle };
    Phase phase = Phase::Live;
    uint64_t last_frame = 0;
    const double file_fps = cap.get(cv::CAP_PROP_FPS); // video file: the pace of the live view
    const double frame_period = 1.0 / (file_fps >= 1 ? file_fps : opt.fps);
    double next_frame = now_s();

    int captures = 0;
    cv::Mat shot; // the captured frame as the camera delivered it
    std::vector<Hand> hands;
    int selected = -1;
    std::optional<status_msg> status; // newest reply that comes after the hand's target was set
    std::string note;                 // what is happening to the command, for the banner
    uint64_t base = 0;                // Link::replies() before this capture's messages went out
    double t_phase = 0, t_poll = 0;   // when the phase began, when the last status request went out
    int on_target = 0;                // status replies in a row with every joint at its target

    // Shows the capture with what is known so far in the "last capture" panel. A small frame is scaled up first,
    // so that the banner and the joint bars do not hide the hand.
    auto redraw = [&]
    {
        const int scale = std::max(1, 480 / shot.rows);
        cv::Mat picture;
        cv::resize(shot, picture, cv::Size(), scale, scale, cv::INTER_LINEAR);
        std::vector<Hand> scaled = hands;
        for (Hand& hand : scaled)
            for (cv::Point2f& p : hand.points)
            {
                p.x *= static_cast<float>(scale);
                p.y *= static_cast<float>(scale);
            }
        std::vector<std::string> lines = {"Capture " + std::to_string(captures) + ": " +
                                              (selected >= 0 ? "hand found" : "no " + opt.hand + " hand found"),
                                          note};
        if (status) lines.push_back(status->text);
        draw_preview(picture, scaled, selected, status ? &*status : nullptr, lines, selected >= 0 && status);
        web.set_shot(picture);
    };

    // The capture is over, or given up on: show how it ended, and keep the button locked for the safety time.
    auto settle = [&](const std::string& outcome, const std::string& page_text)
    {
        note = outcome;
        redraw();
        web.set_status(page_text + " Ready again in " + std::to_string(std::lround(SNAPSHOT_SETTLE)) + " s.");
        std::cout << "Capture " << captures << ": " << outcome << std::endl;
        phase = Phase::Settle;
        t_phase = now_s();
    };

    int exit_code = 0;
    try
    {
        while (!g_stop)
        {
            cv::Mat image;
            double stamp;
            if (browser)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(SNAPSHOT_BROWSER_POLL_MS));
                stamp = now_s();
            }
            else if (grabber)
            {
                const Grabber::Frame frame = grabber->next(last_frame);
                if (frame.count == last_frame) continue; // no new frame within 1 s
                image = frame.image;
                stamp = frame.stamp;
                last_frame = frame.count;
            }
            else
            {
                const double wait = next_frame - now_s();
                if (wait > 0) std::this_thread::sleep_for(std::chrono::duration<double>(wait));
                next_frame = std::max(next_frame + frame_period, now_s()); // no catching up after a long capture
                if (!cap.read(image) && !(cap.set(cv::CAP_PROP_POS_FRAMES, 0) && cap.read(image)))
                    throw std::runtime_error("could not read the video file");
                stamp = now_s();
            }
            if (!browser)
            {
                if (opt.flip_input) cv::flip(image, image, 1);
                cv::Mat live;
                cv::flip(image, live, 1); // mirror view, like the preview
                web.publish(live);
            }
            const double now = now_s();

            if (phase == Phase::Live)
            {
                if (browser)
                {
                    if (!web.take_capture(&image)) continue;
                    if (opt.flip_input) cv::flip(image, image, 1);
                    stamp = now;
                }
                else if (!web.take_capture()) continue;
                ++captures;
                shot = image.clone();
                hands.clear();
                selected = -1;
                status.reset();
                on_target = 0;
                note.clear();
                cv::Mat clicked;
                cv::flip(shot, clicked, 1);
                web.set_shot(clicked); // the image that was clicked, until the landmarks are there

                cv::Mat rgb;
                cv::cvtColor(shot, rgb, cv::COLOR_BGR2RGB);
                const double t0 = now_s();
                hands = landmarker.find(rgb, right);
                selected = select_hand(hands, right);
                const long ms = std::lround((now_s() - t0) * 1e3);

                if (selected < 0)
                {
                    note = "Nothing was sent to the hand";
                    redraw();
                    std::string text = "No " + opt.hand + " hand found";
                    if (!hands.empty()) text += std::string(" (a ") + (right ? "left" : "right") + " hand was seen)";
                    std::cout << "Capture " << captures << ": " << text << " (" << ms << " ms), nothing sent" << std::endl;
                    web.finish(text + ". Nothing was sent to the hand.");
                    continue;
                }

                note = "Sending the command...";
                redraw();
                web.set_status("Hand found. Sending the command to the hand...");
                std::cout << "Capture " << captures << ": " << opt.hand << " hand found (" << ms << " ms)" << std::endl;
                link.newest_reply(); // replies to earlier messages must not count
                base = link.replies();
                // A "no hand" message first, so that the filter in mimic_ctrl starts afresh and the command is the
                // pose as seen, not blended with the last capture. It moves nothing; the hand holds its pose.
                if (!(link.send(make_message(right, now_s(), nullptr)) && link.send(make_message(right, stamp, &hands[selected]))))
                    settle(std::string("Send failed: ") + std::strerror(errno), "Could not send the command to mimic_ctrl.");
                else
                {
                    phase = Phase::Ack;
                    t_phase = now_s();
                }
            }
            else if (phase == Phase::Ack)
            {
                link.newest_reply();
                if (link.replies() >= base + 2) // one reply for each of the two messages: mimic_ctrl has them
                {
                    note = "Moving the hand...";
                    web.set_status("Command sent. Waiting for the hand to move...");
                    phase = Phase::Reach;
                    t_phase = now;
                    t_poll = 0;
                }
                else if (now - t_phase > SNAPSHOT_ACK_TIMEOUT)
                    settle("No reply from mimic_ctrl on port " + std::to_string(opt.port),
                           "No reply from mimic_ctrl: the command may not have reached the hand.");
            }
            else if (phase == Phase::Reach)
            {
                // mimic_ctrl replies before it applies a message's target, so only a reply to a later message
                // (the status requests below) holds the target that was commanded.
                const auto reply = link.newest_reply();
                if (reply && link.replies() >= base + 3)
                {
                    const bool first = !status;
                    status = reply;
                    on_target = at_target(*status) ? on_target + 1 : 0;
                    if (!status->got_feedback) // dry run: there is no hand to wait for
                        settle(status->dry_run ? "Dry run: no hand connected" : "No position feedback from the hand",
                               status->dry_run ? "Dry run: the command was received, no hand is connected."
                                               : "Command sent, but the hand reports no positions to wait for.");
                    else if (on_target >= SNAPSHOT_REACH_POLLS) settle("Hand at target", "The hand reached its target.");
                    else if (first) redraw(); // the targets are known now: show them
                }
                if (phase == Phase::Reach && now - t_phase > SNAPSHOT_REACH_TIMEOUT)
                    settle("Hand did not reach the target", "The hand did not reach its target in time.");
                if (phase == Phase::Reach && now - t_poll >= SNAPSHOT_POLL)
                {
                    link.send(make_message(right, now, nullptr)); // asks for a status; moves nothing
                    t_poll = now;
                }
            }
            else if (now - t_phase >= SNAPSHOT_SETTLE) // Phase::Settle
            {
                phase = Phase::Live;
                web.finish("Ready. Press Capture for the next image.");
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "hand_cam error: " << e.what() << "\n";
        exit_code = 2;
    }

    send_quit(link);
    return exit_code;
}

int main(int argc, char** argv)
{
    Options opt;
    try
    {
        opt = parse_args(argc, argv);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Argument error: " << e.what() << "\n";
        print_usage(argv[0]);
        return 2;
    }
#ifdef HAND_CAM_HEADLESS
    opt.no_preview = true;
#endif
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    try
    {
        return opt.bench ? run_bench(opt) : opt.snapshot ? run_snapshot(opt) : run(opt);
    }
    catch (const std::exception& e) // setup failed: models, camera or socket
    {
        std::cerr << "hand_cam error: " << e.what() << "\n";
        return 2;
    }
}
