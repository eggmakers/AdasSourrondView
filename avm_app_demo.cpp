/***
 * function: 360 surround view combine c++ demo (Real-time Camera Version)
 * author: joker.mao
 * date: 2023/07/15
 * copyright: ADAS_EYES all right reserved
 * modification: Further optimization for undistort and transform
 */

#include "common.h"
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <mutex>
#include <cstring>

// #define DEBUG
#define AWB_LUN_BANLANCE_ENALE 1
// #define DEBUG 0

// 全局变量用于多线程
std::mutex g_mutex;
std::atomic<bool> g_processing_complete(false);

// 预计算映射表结构
struct UndistortMaps
{
    cv::Mat map1;
    cv::Mat map2;
};

// 预计算每个相机的映射表
UndistortMaps precomputeUndistortMaps(const CameraPrms &prm)
{
    UndistortMaps maps;

    // 获取新的相机矩阵
    cv::Mat new_camera_matrix = prm.camera_matrix.clone();
    double *matrix_data = (double *)new_camera_matrix.data;

    const auto scale = (const float *)(prm.scale_xy.data);
    const auto shift = (const float *)(prm.shift_xy.data);

    if (matrix_data && scale && shift)
    {
        matrix_data[0] *= (double)scale[0];
        matrix_data[3 * 1 + 1] *= (double)scale[1];
        matrix_data[2] += (double)shift[0];
        matrix_data[1 * 3 + 2] += (double)shift[1];
    }

    // 预计算映射表
    cv::fisheye::initUndistortRectifyMap(prm.camera_matrix, prm.dist_coff, cv::Mat(),
                                         new_camera_matrix, prm.size, CV_16SC2,
                                         maps.map1, maps.map2);

    return maps;
}

// 多线程处理单个摄像头图像的函数
void processCameraImage(cv::Mat &src, cv::Mat &dst, const CameraPrms &prm,
                        const UndistortMaps &maps, const char *flip_mir)
{
    try
    {
        cv::Mat temp_src = src.clone();

        // 使用预计算的映射表进行畸变校正
        cv::remap(temp_src, temp_src, maps.map1, maps.map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

        // 透视变换
        cv::warpPerspective(temp_src, temp_src, prm.project_matrix, project_shapes[prm.name]);

        // 旋转处理 - 使用 C 风格字符串比较
        if (strcmp(flip_mir, "r+") == 0)
        {
            cv::rotate(temp_src, temp_src, cv::ROTATE_90_CLOCKWISE);
        }
        else if (strcmp(flip_mir, "r-") == 0)
        {
            cv::rotate(temp_src, temp_src, cv::ROTATE_90_COUNTERCLOCKWISE);
        }
        else if (strcmp(flip_mir, "m") == 0)
        {
            cv::rotate(temp_src, temp_src, cv::ROTATE_180);
        }

        std::lock_guard<std::mutex> lock(g_mutex);
        temp_src.copyTo(dst);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error processing camera image: " << e.what() << std::endl;
    }
}

// 优化版本：合并畸变校正和透视变换
void optimizedProcessCameraImage(cv::Mat &src, cv::Mat &dst, const CameraPrms &prm,
                                 const UndistortMaps &maps, const char *flip_mir)
{
    try
    {
        // 使用预计算的映射表进行畸变校正
        cv::Mat undistorted;
        cv::remap(src, undistorted, maps.map1, maps.map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

        // 透视变换
        cv::Mat transformed;
        cv::warpPerspective(undistorted, transformed, prm.project_matrix, project_shapes[prm.name]);

        // 旋转处理
        cv::Mat rotated;
        if (strcmp(flip_mir, "r+") == 0)
        {
            cv::rotate(transformed, rotated, cv::ROTATE_90_CLOCKWISE);
        }
        else if (strcmp(flip_mir, "r-") == 0)
        {
            cv::rotate(transformed, rotated, cv::ROTATE_90_COUNTERCLOCKWISE);
        }
        else if (strcmp(flip_mir, "m") == 0)
        {
            cv::rotate(transformed, rotated, cv::ROTATE_180);
        }
        else
        {
            rotated = transformed;
        }

        std::lock_guard<std::mutex> lock(g_mutex);
        rotated.copyTo(dst);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error processing camera image: " << e.what() << std::endl;
    }
}

void processAllCamerasParallel(cv::Mat *origin_dir_img, cv::Mat *undist_dir_img,
                               CameraPrms *prms, const UndistortMaps *maps,
                               const char *camera_flip_mir[4])
{
    std::vector<std::thread> threads;

    for (int i = 0; i < 4; ++i)
    {
        threads.emplace_back(optimizedProcessCameraImage,
                             std::ref(origin_dir_img[i]),
                             std::ref(undist_dir_img[i]),
                             std::cref(prms[i]),
                             std::cref(maps[i]),
                             camera_flip_mir[i]);
    }

    for (auto &t : threads)
    {
        if (t.joinable())
        {
            t.join();
        }
    }
}

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
    cv::Mat weights_complement_img[4];
    cv::Mat out_put_img;
    float *w_ptr[4];
    CameraPrms prms[4];
    UndistortMaps undistort_maps[4]; // 预计算的映射表
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
        weights_complement_img[i] = cv::Mat(weights.size(), CV_32FC1, cv::Scalar(0, 0, 0));
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

    // 计算权重补矩阵 (1 - weights)
    for (int i = 0; i < 4; ++i)
    {
        cv::subtract(cv::Scalar(1.0), merge_weights_img[i], weights_complement_img[i]);
    }

#ifdef DEBUG
    for (int i = 0; i < 4; ++i)
    {
        display_mat(merge_weights_img[i], "w");
        display_mat(weights_complement_img[i], "w_complement");
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

        // 预计算畸变校正映射表
        undistort_maps[i] = precomputeUndistortMaps(prm);
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

    // 性能统计变量
    double total_frame_read_time = 0.0;
    double total_awb_time = 0.0;
    double total_undistort_time = 0.0;
    double total_stitching_time = 0.0;
    double total_frame_time = 0.0;

    // 创建 UMat 变量用于权重和权重补矩阵
    cv::UMat weights_umat[4];
    cv::UMat weights_complement_umat[4];

    for (int i = 0; i < 4; ++i)
    {
        merge_weights_img[i].copyTo(weights_umat[i]);
        weights_complement_img[i].copyTo(weights_complement_umat[i]);
    }

    // 预分配内存用于处理结果
    for (int i = 0; i < 4; ++i)
    {
        undist_dir_img[i] = cv::Mat(project_shapes[prms[i].name], CV_8UC3);
    }

    while (running)
    {
        auto frame_start_time = std::chrono::high_resolution_clock::now();
        double frame_read_time = 0.0;
        double awb_time = 0.0;
        double undistort_time = 0.0;
        double stitching_time = 0.0;

        // 1. 读取帧（计时开始）
        auto frame_read_start = std::chrono::high_resolution_clock::now();

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

        auto frame_read_end = std::chrono::high_resolution_clock::now();
        frame_read_time = std::chrono::duration<double, std::milli>(frame_read_end - frame_read_start).count();

        frame_count++;
        fps_frame_count++;

        // 2. 白平衡和亮度均衡（计时开始）
        auto awb_start = std::chrono::high_resolution_clock::now();

        std::vector<cv::Mat *> srcs;
        for (int i = 0; i < 4; ++i)
        {
            srcs.push_back(&origin_dir_img[i]);
        }

#if AWB_LUN_BANLANCE_ENALE
        awb_and_lum_banlance(srcs);
#endif

        auto awb_end = std::chrono::high_resolution_clock::now();
        awb_time = std::chrono::duration<double, std::milli>(awb_end - awb_start).count();

        // 3. 畸变校正和图像变换 - 使用多线程优化（计时开始）
        auto undistort_start = std::chrono::high_resolution_clock::now();

        // 使用多线程并行处理四个摄像头
        processAllCamerasParallel(origin_dir_img, undist_dir_img, prms, undistort_maps, camera_flip_mir);

        auto undistort_end = std::chrono::high_resolution_clock::now();
        undistort_time = std::chrono::duration<double, std::milli>(undistort_end - undistort_start).count();

        // 4. 图像拼接（计时开始）
        auto stitching_start = std::chrono::high_resolution_clock::now();

        out_put_img.setTo(cv::Scalar(0, 0, 0));
        car_img.copyTo(out_put_img(cv::Rect(xl, yt, car_img.cols, car_img.rows)));

        // 4.1 中心区域复制
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

        // 4.2 四个角融合 - 使用优化后的 merge_image 函数
        cv::Rect roi;
        // 左上角
        roi = cv::Rect(0, 0, xl, yt);
        {
            cv::UMat src1_umat, src2_umat, out_umat;
            undist_dir_img[0](roi).copyTo(src1_umat);
            undist_dir_img[1](roi).copyTo(src2_umat);
            out_put_img(roi).copyTo(out_umat);

            merge_image(src1_umat, src2_umat, weights_umat[2], weights_complement_umat[2], out_umat);

            out_umat.copyTo(out_put_img(roi));
        }

        // 右上角
        roi = cv::Rect(xr, 0, xl, yt);
        {
            cv::UMat src1_umat, src2_umat, out_umat;
            undist_dir_img[0](roi).copyTo(src1_umat);
            undist_dir_img[3](cv::Rect(0, 0, xl, yt)).copyTo(src2_umat);
            out_put_img(roi).copyTo(out_umat);

            merge_image(src1_umat, src2_umat, weights_umat[1], weights_complement_umat[1], out_umat);

            out_umat.copyTo(out_put_img(roi));
        }

        // 左下角
        roi = cv::Rect(0, yb, xl, yt);
        {
            cv::UMat src1_umat, src2_umat, out_umat;
            undist_dir_img[2](cv::Rect(0, 0, xl, yt)).copyTo(src1_umat);
            undist_dir_img[1](roi).copyTo(src2_umat);
            out_put_img(roi).copyTo(out_umat);

            merge_image(src1_umat, src2_umat, weights_umat[0], weights_complement_umat[0], out_umat);

            out_umat.copyTo(out_put_img(roi));
        }

        // 右下角
        roi = cv::Rect(xr, yb, xl, yt);
        {
            cv::UMat src1_umat, src2_umat, out_umat;
            undist_dir_img[2](cv::Rect(xr, 0, xl, yt)).copyTo(src1_umat);
            undist_dir_img[3](cv::Rect(0, yb, xl, yt)).copyTo(src2_umat);
            out_put_img(roi).copyTo(out_umat);

            merge_image(src1_umat, src2_umat, weights_umat[3], weights_complement_umat[3], out_umat);

            out_umat.copyTo(out_put_img(roi));
        }

        auto stitching_end = std::chrono::high_resolution_clock::now();
        stitching_time = std::chrono::duration<double, std::milli>(stitching_end - stitching_start).count();

        // 计算总帧时间
        auto frame_end_time = std::chrono::high_resolution_clock::now();
        double total_frame_time_current = std::chrono::duration<double, std::milli>(frame_end_time - frame_start_time).count();

        // 累加性能统计
        total_frame_read_time += frame_read_time;
        total_awb_time += awb_time;
        total_undistort_time += undistort_time;
        total_stitching_time += stitching_time;
        total_frame_time += total_frame_time_current;

        // 计算帧率
        auto current_time = std::chrono::high_resolution_clock::now();
        auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_fps_time).count();

        // 每秒更新一次FPS显示和性能统计
        if (time_diff > 1000)
        {
            current_fps = fps_frame_count * 1000.0 / time_diff;
            fps_frame_count = 0;
            last_fps_time = current_time;

            // 输出性能统计到控制台
            std::cout << "Current FPS: " << current_fps << " FPS" << std::endl;
            std::cout << "Stage Timings (ms):" << std::endl;
            std::cout << "  Frame Read: " << total_frame_read_time / frame_count << std::endl;
            std::cout << "  AWB & Luminance: " << total_awb_time / frame_count << std::endl;
            std::cout << "  Undistort & Transform: " << total_undistort_time / frame_count << std::endl;
            std::cout << "  Image Stitching: " << total_stitching_time / frame_count << std::endl;
            std::cout << "  Total Frame Time: " << total_frame_time / frame_count << std::endl;
        }

        // 在图像上显示FPS信息
        std::string fps_text = "FPS: " + std::to_string(static_cast<int>(current_fps));
        std::string frame_text = "Frames: " + std::to_string(frame_count);

        int font_face = cv::FONT_HERSHEY_SIMPLEX;
        double font_scale = 0.7;
        int thickness = 2;
        cv::Scalar text_color(0, 255, 0);
        cv::Scalar bg_color(0, 0, 0);

        int baseline = 0;
        cv::Size fps_text_size = cv::getTextSize(fps_text, font_face, font_scale, thickness, &baseline);
        cv::Size frame_text_size = cv::getTextSize(frame_text, font_face, font_scale, thickness, &baseline);

        int padding = 5;
        cv::Point text_org(10, 30);

        // 绘制半透明背景
        cv::Rect bg_rect(text_org.x - padding, text_org.y - fps_text_size.height - padding,
                         std::max(fps_text_size.width, frame_text_size.width) + 2 * padding,
                         fps_text_size.height + frame_text_size.height + 3 * padding);
        cv::Mat roi_bg = out_put_img(bg_rect);
        cv::Mat bg_overlay(roi_bg.size(), roi_bg.type(), bg_color);
        cv::addWeighted(bg_overlay, 0.3, roi_bg, 0.7, 0, roi_bg);

        // 绘制FPS文本
        cv::putText(out_put_img, fps_text, text_org, font_face, font_scale, text_color, thickness);
        cv::putText(out_put_img, frame_text, cv::Point(text_org.x, text_org.y + fps_text_size.height + 10),
                    font_face, font_scale, text_color, thickness);

        // 显示结果
        cv::imshow("360 Surround View", out_put_img);

        // 处理键盘输入
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

    // 计算总平均帧率
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    double average_fps = (total_duration > 0) ? frame_count * 1000.0 / total_duration : 0;

    std::cout << "Program statistics:" << std::endl;
    std::cout << "Total frames: " << frame_count << std::endl;
    std::cout << "Total runtime: " << total_duration / 1000.0 << " seconds" << std::endl;
    std::cout << "Average FPS: " << average_fps << " FPS" << std::endl;

    // 输出最终性能统计
    if (frame_count > 0)
    {
        std::cout << "Final Performance Statistics (average per frame):" << std::endl;
        std::cout << "  Frame Read: " << total_frame_read_time / frame_count << " ms" << std::endl;
        std::cout << "  AWB & Luminance: " << total_awb_time / frame_count << " ms" << std::endl;
        std::cout << "  Undistort & Transform: " << total_undistort_time / frame_count << " ms" << std::endl;
        std::cout << "  Image Stitching: " << total_stitching_time / frame_count << " ms" << std::endl;
        std::cout << "  Total Frame Time: " << total_frame_time / frame_count << " ms" << std::endl;
    }

    // 释放资源
    for (int i = 0; i < 4; ++i)
    {
        caps[i].release();
    }
    cv::destroyAllWindows();

    std::cout << argv[0] << " app finished. Processed " << frame_count << " frames." << std::endl;
    return 0;
}