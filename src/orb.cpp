#include "orb.hpp"

#include <array>
#include <iostream>
#include <utility>
#include <vector>

#define WIDTH 640
#define HEIGHT 640

// template <typename T>
// std::vector<T> brensenham(const cv::Mat& img, int row, int col, int radius) {
//     // Brensenham Circle algorithm on an image, reference:
//     // https://www.geeksforgeeks.org/mid-point-circle-drawing-algorithm/
//     vector<T> v;
//     int P = 1 - radius;
//     int img_height = img.cols;
//     int img_width = img.rows;
//     while (x > y) {
//         y++;
//         if (P <= 0) {
//             P = P + 2*y + 1;
//         } else {
//             x--;
//             P = P + 2 * y - 2 * x + 1;
//         }
//     if (x < y) break;
//     }
// }

// // Can make this optimized with GPU, but premature optimization is the root of all evil.
// void search() {
//     for (int i = 0; i < 2 * n; i++) {
//         if (values[i % n] > pos_thresh) {
//             contiguous_count++;
//         } else {
//             contiguous_count = 0;
//         }
//         int curr = positions[i % n];
//         longest = max(longest, curr);
//     }
// }

void process(const cv::Mat& img) {
    // Paper: https://ieeexplore.ieee.org/document/6126544
    // Steps
    // 1. Detect ORB features
    // 2.
    const int radius = 9;
    const int threshold = 25;

    double t = (double)cv::getTickCount();
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

    // Automatically runs multithreading
    // reference: https://stackoverflow.com/questions/4504687/cycle-through-pixels-with-opencv
    // Metods from this tutorial are too slow
    // https://docs.opencv.org/4.x/db/da5/tutorial_how_to_scan_images.html

    // TODO: check what DS to use here instead of vectors of points
    // A corner is a feature in "FAST" speak
    std::vector<std::pair<int, int>> positive_corners;
    std::vector<std::pair<int, int>> negative_corners;

    img.forEach<uchar>([&img, &radius](uchar& p, const int position[]) -> void {
        if (position[0] - radius < 0 || position[0] + radius >= HEIGHT ||
            position[1] - radius < 0 || position[1] + radius >= WIDTH) {
            return;
        }
        // Within
        std::vector<std::pair<int, int>> positive_bordering_pixels;
        std::vector<std::pair<int, int>> negative_bordering_pixels;

        // Check for positive corner
        int count = 0;
        int positive_thresh = (int)p + threshold;

        std::array<int, 4> four_corners = {
            img.ptr<uchar>(position[0] - radius)[position[1]],
            img.ptr<uchar>(position[0] - radius)[position[1]],
            img.ptr<uchar>(position[0])[position[1] + radius],
            img.ptr<uchar>(position[0])[position[1] - radius],
        };

        if (positive_thresh <= 255) {
            // Optimization: Check 4 corner pixels first
            for (int i = 0; i < 4; i++) {
                if (four_corners[i] > positive_thresh)
                    ;
                positive_bordering_pixels.push_back({position[0], position[1]});
            }
            if (positive_bordering_pixels.size() >= 3) {
                // full check for contiguous values
                std::vector<int> values;

                int longest = 0;
                int n = values.size();
                int contiguous_count = 0;
                return;
            }
        }
        int pos_thresh = (int)p + threshold;

        // check for negative corner
        int neg_thresh = (int)p + threshold;
        if (neg_thresh >= 0) {
        }
    });

    cv::imwrite("bw_img.png", img);

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << " processing took: " << t << "s" << std::endl;
}
