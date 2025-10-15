/***
 * function: 360 surrond view combine c++ demo
 * author: joker.mao
 * date: 2023/07/15
 * copyright: ADAS_EYES all right reserved
 */

#include "common.h"

void display_mat(cv::Mat &img, std::string name)
{
    cv::imshow(name, img);
    cv::waitKey();
}

// read cali prms
bool read_prms(const std::string &path, CameraPrms &prms)
{
    cv::FileStorage fs(path, cv::FileStorage::READ);

    if (!fs.isOpened())
    {
        throw std::string("error open file");
        return false;
    }

    prms.camera_matrix = fs["camera_matrix"].mat();
    prms.dist_coff = fs["dist_coeffs"].mat();
    prms.project_matrix = fs["project_matrix"].mat();
    prms.shift_xy = fs["shift_xy"].mat();
    prms.scale_xy = fs["scale_xy"].mat();
    auto size_ = fs["resolution"].mat();
    prms.size = cv::Size(size_.at<int>(0), size_.at<int>(1));

    fs.release();

    return true;
}
// save cali prms
bool save_prms(const std::string &path, CameraPrms &prms)
{
    cv::FileStorage fs(path, cv::FileStorage::WRITE);

    if (!fs.isOpened())
    {
        throw std::string("error open file");
        return false;
    }

    if (!prms.project_matrix.empty())
        fs << "project_matrix" << prms.project_matrix;

    fs.release();

    return true;
}

// undist image by remap
void undist_by_remap(const cv::Mat &src, cv::Mat &dst, const CameraPrms &prms)
{
    // get new camera matrix
    cv::Mat new_camera_matrix = prms.camera_matrix.clone();
    double *matrix_data = (double *)new_camera_matrix.data;

    const auto scale = (const float *)(prms.scale_xy.data);
    const auto shift = (const float *)(prms.shift_xy.data);

    if (!matrix_data || !scale || !shift)
    {
        return;
    }

    matrix_data[0] *= (double)scale[0];
    matrix_data[3 * 1 + 1] *= (double)scale[1];
    matrix_data[2] += (double)shift[0];
    matrix_data[1 * 3 + 2] += (double)shift[1];
    // std::cout << new_camera_matrix;
    // undistort
    cv::Mat map1, map2;
    cv::fisheye::initUndistortRectifyMap(prms.camera_matrix, prms.dist_coff, cv::Mat(), new_camera_matrix, prms.size, CV_16SC2, map1, map2);

    cv::remap(src, dst, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
}

// merge image by weights
void merge_image(cv::UMat src1, cv::UMat src2, cv::UMat weights, cv::UMat weights_complement, cv::UMat out)
{
    cv::UMat weights_3ch;
    cv::merge(std::vector<cv::UMat>{weights, weights, weights}, weights_3ch);

    cv::UMat weights_3ch_complement;
    cv::merge(std::vector<cv::UMat>{weights_complement, weights_complement, weights_complement}, weights_3ch_complement);

    cv::UMat src1_float, src2_float;
    src1.convertTo(src1_float, CV_32FC3);
    src2.convertTo(src2_float, CV_32FC3);

    cv::UMat src1_weighted, src2_weighted;
    cv::multiply(src1_float, weights_3ch, src1_weighted);
    cv::multiply(src2_float, weights_3ch_complement, src2_weighted);

    cv::UMat blended_float;
    cv::add(src1_weighted, src2_weighted, blended_float);

    blended_float.convertTo(out, CV_8UC3);
}

// r g b channel statics
void rgb_info_statics(cv::Mat &src, BgrSts &sts)
{
    if (src.empty())
    {
        sts.b = sts.g = sts.r = 0;
        return;
    }

    // 使用 OpenCV 的 mean 函数计算每个通道的平均值
    cv::Scalar mean = cv::mean(src);
    sts.b = static_cast<int>(mean[0]);
    sts.g = static_cast<int>(mean[1]);
    sts.r = static_cast<int>(mean[2]);
}

// r g b digtial gain
void rgb_dgain(cv::Mat &src, float r_gain, float g_gain, float b_gain)
{
    if (src.empty())
    {
        return;
    }

    // 使用 OpenCV 的分通道处理和乘法
    std::vector<cv::Mat> channels;
    cv::split(src, channels);

    // 对每个通道应用增益
    channels[0] *= b_gain; // Blue channel
    channels[1] *= g_gain; // Green channel
    channels[2] *= r_gain; // Red channel

    // 合并通道
    cv::merge(channels, src);

    // 确保值在 0-255 范围内
    cv::threshold(src, src, 255, 255, cv::THRESH_TRUNC);
}

// gray world awb amd lum banlance for four channeal images
void awb_and_lum_banlance(std::vector<cv::Mat *> srcs)
{
    BgrSts sts[4];
    int gray[4] = {0, 0, 0, 0};
    float gray_ave = 0;

    if (srcs.size() != 4)
    {
        return;
    }

    for (int i = 0; i < 4; ++i)
    {
        if (srcs[i] == nullptr)
        {
            return;
        }
        rgb_info_statics(*srcs[i], sts[i]);
        gray[i] = sts[i].r * 20 + sts[i].g * 60 + sts[i].b;
        gray_ave += gray[i];
    }

    gray_ave /= 4;

    for (int i = 0; i < 4; ++i)
    {
        float lum_gain = gray_ave / gray[i];
        float r_gain = sts[i].g * lum_gain / sts[i].r;
        float g_gain = lum_gain;
        float b_gain = sts[i].g * lum_gain / sts[i].b;
        // std::cout << "gains : " << r_gain << " | " << g_gain << " | " << b_gain << "\r\n";
        rgb_dgain(*srcs[i], r_gain, g_gain, b_gain);
    }
}