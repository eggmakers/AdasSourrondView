/***
 * function: 360 surround view combine c++ demo (Real-time Camera Version)
 * author: joker.mao
 * date: 2023/07/15
 * copyright: ADAS_EYES all right reserved
 * modification: Changed to real-time camera input
 */

#include "common.h"
#include <vector>
#include <thread>
#include <chrono>

// #define DEBUG
#define AWB_LUN_BANLANCE_ENALE 1
// #define DEBUG 0

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cout << "usage:\n\t" << argv[0] << " path\n";
        return -1;
    }
    std::cout << argv[0] << " app start running..." << std::endl;
    std::string data_path = std::string(argv[1]);
    cv::Mat car_img;
    cv::Mat origin_dir_img[4];
    cv::Mat undist_dir_img[4];
    cv::Mat merge_weights_img[4];
    cv::Mat out_put_img;
    float *w_ptr[4];
    CameraPrms prms[4];
    int capture_width = 960;
    int capture_height = 640;

    // 1. Read vehicle image and weight map
    car_img = cv::imread(data_path + "/images/car.png");
    if (car_img.empty())
    {
        std::cerr << "Cannot read vehicle image: " << data_path + "/images/car.png" << std::endl;
        return -1;
    }
    cv::resize(car_img, car_img, cv::Size(xr - xl, yb - yt));
    out_put_img = cv::Mat(cv::Size(total_w, total_h), CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat weights = cv::imread(data_path + "/yaml/weights.png", -1);

    if (weights.channels() != 4)
    {
        std::cerr << "Failed to read weights image, channels: " << weights.channels() << "\r\n";
        return -1;
    }

    for (int i = 0; i < 4; ++i)
    {
        merge_weights_img[i] = cv::Mat(weights.size(), CV_32FC1, cv::Scalar(0, 0, 0));
        w_ptr[i] = (float *)merge_weights_img[i].data;
    }

    // Read weight map data
    int pixel_index = 0;
    for (int h = 0; h < weights.rows; ++h)
    {
        uchar *uc_pixel = weights.data + h * weights.step;
        for (int w = 0; w < weights.cols; ++w)
        {
            w_ptr[0][pixel_index] = uc_pixel[0] / 255.0f;
            w_ptr[1][pixel_index] = uc_pixel[1] / 255.0f;
            w_ptr[2][pixel_index] = uc_pixel[2] / 255.0f;
            w_ptr[3][pixel_index] = uc_pixel[3] / 255.0f;
            uc_pixel += 4;
            ++pixel_index;
        }
    }

#ifdef DEBUG
    for (int i = 0; i < 4; ++i)
    {
        display_mat(merge_weights_img[i], "w");
    }
#endif

    // Read camera calibration parameters
    for (int i = 0; i < 4; ++i)
    {
        auto &prm = prms[i];
        prm.name = camera_names[i];
        auto ok = read_prms(data_path + "/yaml/" + prm.name + ".yaml", prm);
        if (!ok)
        {
            return -1;
        }
    }

    // Initialize four cameras
    cv::VideoCapture caps[4];
    std::string camera_devices[4] = {"/dev/video44", "/dev/video71", "/dev/video53", "/dev/video62"};
    std::string gstreamer_pipelines[4] =
        {
            "v4l2src device=/dev/video44 ! "
            "video/x-raw, format=UYVY, width=960, height=640 ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! "
            "appsink sync=false",
            "v4l2src device=/dev/video71 ! "
            "video/x-raw, format=UYVY, width=960, height=640 ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! "
            "appsink sync=false",
            "v4l2src device=/dev/video53 ! "
            "video/x-raw, format=UYVY, width=960, height=640 ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! "
            "appsink sync=false",
            "v4l2src device=/dev/video62 ! "
            "video/x-raw, format=UYVY, width=960, height=640 ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! "
            "appsink sync=false"};

    for (int i = 0; i < 4; ++i)
    {
        // Try GStreamer pipeline first
        std::cout << "Attempting to open camera " << camera_names[i] << " with GStreamer" << std::endl;
        caps[i] = cv::VideoCapture(gstreamer_pipelines[i], cv::CAP_GSTREAMER);

        if (!caps[i].isOpened())
        {
            std::cerr << "Error: Cannot open camera " << camera_names[i] << std::endl;
            return -1;
        }

        std::cout << "Successfully opened camera " << camera_names[i]
                  << " Resolution: " << caps[i].get(cv::CAP_PROP_FRAME_WIDTH)
                  << "x" << caps[i].get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
    }

    // Real-time processing loop
    std::cout << "Starting real-time 360 surround view processing..." << std::endl;
    std::cout << "Press 'q' to exit program" << std::endl;
    std::cout << "Press 's' to save current frame" << std::endl;

    bool running = true;
    int frame_count = 0;

    // Frame rate calculation variables
    auto start_time = std::chrono::high_resolution_clock::now();
    auto last_fps_time = start_time;
    int fps_frame_count = 0;
    double current_fps = 0.0;

    while (running)
    {
        auto frame_start_time = std::chrono::high_resolution_clock::now();

        // Read frames from four cameras
        bool all_frames_ready = true;
        for (int i = 0; i < 4; ++i)
        {
            if (!caps[i].read(origin_dir_img[i]))
            {
                std::cerr << "Failed to read frame from camera: " << camera_names[i] << std::endl;
                all_frames_ready = false;
                break;
            }
        }

        if (!all_frames_ready)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        frame_count++;
        fps_frame_count++;

        // 2. White balance and luminance balancing
        std::vector<cv::Mat *> srcs;
        for (int i = 0; i < 4; ++i)
        {
            srcs.push_back(&origin_dir_img[i]);
        }

#if AWB_LUN_BANLANCE_ENALE
        awb_and_lum_banlance(srcs);
#endif

        // 3. Distortion removal and image transformation
        for (int i = 0; i < 4; ++i)
        {
            auto &prm = prms[i];
            cv::Mat &src = origin_dir_img[i];

            undist_by_remap(src, src, prm);
            cv::warpPerspective(src, src, prm.project_matrix, project_shapes[prm.name]);

            if (camera_flip_mir[i] == "r+")
            {
                cv::rotate(src, src, cv::ROTATE_90_CLOCKWISE);
            }
            else if (camera_flip_mir[i] == "r-")
            {
                cv::rotate(src, src, cv::ROTATE_90_COUNTERCLOCKWISE);
            }
            else if (camera_flip_mir[i] == "m")
            {
                cv::rotate(src, src, cv::ROTATE_180);
            }
            undist_dir_img[i] = src.clone();
        }

        // 4. Image stitching
        out_put_img.setTo(cv::Scalar(0, 0, 0));
        car_img.copyTo(out_put_img(cv::Rect(xl, yt, car_img.cols, car_img.rows)));

        // 4.1 Center region copy
        for (int i = 0; i < 4; ++i)
        {
            cv::Rect roi;
            if (std::string(camera_names[i]) == "front")
            {
                roi = cv::Rect(xl, 0, xr - xl, yt);
                undist_dir_img[i](roi).copyTo(out_put_img(roi));
            }
            else if (std::string(camera_names[i]) == "left")
            {
                roi = cv::Rect(0, yt, xl, yb - yt);
                undist_dir_img[i](roi).copyTo(out_put_img(roi));
            }
            else if (std::string(camera_names[i]) == "right")
            {
                roi = cv::Rect(0, yt, xl, yb - yt);
                undist_dir_img[i](roi).copyTo(out_put_img(cv::Rect(xr, yt, total_w - xr, yb - yt)));
            }
            else if (std::string(camera_names[i]) == "back")
            {
                roi = cv::Rect(xl, 0, xr - xl, yt);
                undist_dir_img[i](roi).copyTo(out_put_img(cv::Rect(xl, yb, xr - xl, yt)));
            }
        }

        // 4.2 Four corner blending
        cv::Rect roi;
        // Top-left corner
        roi = cv::Rect(0, 0, xl, yt);
        merge_image(undist_dir_img[0](roi), undist_dir_img[1](roi), merge_weights_img[2], out_put_img(roi));
        // Top-right corner
        roi = cv::Rect(xr, 0, xl, yt);
        merge_image(undist_dir_img[0](roi), undist_dir_img[3](cv::Rect(0, 0, xl, yt)), merge_weights_img[1], out_put_img(cv::Rect(xr, 0, xl, yt)));
        // Bottom-left corner
        roi = cv::Rect(0, yb, xl, yt);
        merge_image(undist_dir_img[2](cv::Rect(0, 0, xl, yt)), undist_dir_img[1](roi), merge_weights_img[0], out_put_img(roi));
        // Bottom-right corner
        roi = cv::Rect(xr, 0, xl, yt);
        merge_image(undist_dir_img[2](roi), undist_dir_img[3](cv::Rect(0, yb, xl, yt)), merge_weights_img[3], out_put_img(cv::Rect(xr, yb, xl, yt)));

        // Calculate frame rate
        auto current_time = std::chrono::high_resolution_clock::now();
        auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_fps_time).count();

        // Update FPS display every second
        if (time_diff > 1000)
        {
            current_fps = fps_frame_count * 1000.0 / time_diff;
            fps_frame_count = 0;
            last_fps_time = current_time;

            // Output FPS info to console
            std::cout << "Current FPS: " << current_fps << " FPS" << std::endl;
        }

        // Display FPS information on image
        std::string fps_text = "FPS: " + std::to_string(static_cast<int>(current_fps));
        std::string frame_text = "Frames: " + std::to_string(frame_count);

        // Set text properties
        int font_face = cv::FONT_HERSHEY_SIMPLEX;
        double font_scale = 0.7;
        int thickness = 2;
        cv::Scalar text_color(0, 255, 0); // Green text
        cv::Scalar bg_color(0, 0, 0);     // Black background

        // Get text size
        int baseline = 0;
        cv::Size fps_text_size = cv::getTextSize(fps_text, font_face, font_scale, thickness, &baseline);
        cv::Size frame_text_size = cv::getTextSize(frame_text, font_face, font_scale, thickness, &baseline);

        // Display FPS info at top-left corner with background for better readability
        int padding = 5;
        cv::Point text_org(10, 30);

        // Draw semi-transparent background
        cv::Rect bg_rect(text_org.x - padding, text_org.y - fps_text_size.height - padding,
                         std::max(fps_text_size.width, frame_text_size.width) + 2 * padding,
                         fps_text_size.height + frame_text_size.height + 3 * padding);
        cv::Mat roi_bg = out_put_img(bg_rect);
        cv::Mat bg_overlay(roi_bg.size(), roi_bg.type(), bg_color);
        cv::addWeighted(bg_overlay, 0.3, roi_bg, 0.7, 0, roi_bg);

        // Draw FPS text
        cv::putText(out_put_img, fps_text, text_org, font_face, font_scale, text_color, thickness);
        cv::putText(out_put_img, frame_text, cv::Point(text_org.x, text_org.y + fps_text_size.height + 10),
                    font_face, font_scale, text_color, thickness);

        // Display result
        cv::imshow("360 Surround View", out_put_img);

        // Handle keyboard input
        char key = cv::waitKey(1);
        if (key == 'q' || key == 'Q')
        {
            running = false;
            std::cout << "User requested exit..." << std::endl;
        }
        else if (key == 's' || key == 'S')
        {
            std::string filename = "360_view_frame_" + std::to_string(frame_count) + ".png";
            cv::imwrite(filename, out_put_img);
            std::cout << "Frame saved to: " << filename << std::endl;
        }
    }

    // Calculate total average frame rate
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    double average_fps = (total_duration > 0) ? frame_count * 1000.0 / total_duration : 0;

    std::cout << "Program statistics:" << std::endl;
    std::cout << "Total frames: " << frame_count << std::endl;
    std::cout << "Total runtime: " << total_duration / 1000.0 << " seconds" << std::endl;
    std::cout << "Average FPS: " << average_fps << " FPS" << std::endl;

    // Release resources
    for (int i = 0; i < 4; ++i)
    {
        caps[i].release();
    }
    cv::destroyAllWindows();

    std::cout << argv[0] << " app finished. Processed " << frame_count << " frames." << std::endl;
    return 0;
}