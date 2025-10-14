#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    // 构建GStreamer管道字符串
    std::string pipeline = "v4l2src device=/dev/video44 ! "
                           "video/x-raw, format=UYVY, width=1920, height=1080 ! "
                           "videoconvert ! "
                           "appsink sync=false"; // 注意：将ximagesink替换为appsink

    // 使用CAP_GSTREAMER打开管道
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);

    if (!cap.isOpened())
    {
        std::cerr << "错误：无法打开GStreamer管道。" << std::endl;
        std::cerr << "请检查：\n"
                  << "1. 设备/dev/video11是否存在且权限正确。\n"
                  << "2. OpenCV是否已编译GStreamer支持。\n"
                  << "3. 管道语法是否正确。" << std::endl;
        return -1;
    }

    cv::Mat frame;
    std::string windowName = "GStreamer + OpenCV";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    std::cout << "成功开启视频流。按 'q' 键退出。" << std::endl;

    while (true)
    {
        cap >> frame; // 从管道中读取一帧
        if (frame.empty())
        {
            std::cerr << "错误：捕获到空帧。" << std::endl;
            break;
        }

        cv::imshow(windowName, frame); // 显示帧

        // 等待1毫秒，并检查是否按下'q'键
        char key = (char)cv::waitKey(1);
        if (key == 'q' || key == 27)
        { // 'q' 或 ESC 键
            break;
        }
    }

    // 清理资源
    cap.release();
    cv::destroyAllWindows();
    std::cout << "程序已退出。" << std::endl;
    return 0;
}
