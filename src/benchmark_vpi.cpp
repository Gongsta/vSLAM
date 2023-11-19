#include <vpi/Event.h>
#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/GaussianFilter.h>

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <vector>

#include "vpi_utils.hpp"

int main(int argc, char *argv[]) {
  VPIImage image = NULL;
  VPIImage blurred = NULL;
  VPIStream stream = NULL;

  VPIEvent evStart = NULL;
  VPIEvent evStop = NULL;

  int retval = 0;

  try {
    // 1. Processing of command line parameters -----------

    if (argc != 2) {
      throw std::runtime_error(std::string("Usage: ") + argv[0] + " <cpu|pva|cuda>");
    }

    std::string strBackend = argv[1];

    // Parse the backend
    VPIBackend backend;

    if (strBackend == "cpu") {
      backend = VPI_BACKEND_CPU;
    } else if (strBackend == "cuda") {
      backend = VPI_BACKEND_CUDA;
    } else if (strBackend == "pva") {
      backend = VPI_BACKEND_PVA;
    } else {
      throw std::runtime_error("Backend '" + strBackend +
                               "' not recognized, it must be either cpu, cuda or pva.");
    }

    // 2. Initialization stage ----------------------

    // Create the stream for the given backend.
    CHECK_STATUS(vpiStreamCreate(0, &stream));

    int width = 1920, height = 1080;
    VPIImageFormat imgFormat = VPI_IMAGE_FORMAT_U16;

    std::cout << "Input size: " << width << " x " << height << '\n'
              << "Image format: " << vpiImageFormatGetName(imgFormat) << '\n'
              << "Algorithm: 5x5 Gaussian Filter" << std::endl;

    // Memory flags set to guarantee top performance.
    // Only the benchmarked backend is enabled, and memories
    // are guaranteed to be used by only one stream.
    uint64_t memFlags = backend | VPI_EXCLUSIVE_STREAM_ACCESS;

    // Create image with zero content
    CHECK_STATUS(vpiImageCreate(width, height, imgFormat, memFlags, &image));

    // Create a temporary image convolved with a low-pass filter.
    CHECK_STATUS(vpiImageCreate(width, height, imgFormat, memFlags, &blurred));

    // Create the events we'll need to get timing info
    CHECK_STATUS(vpiEventCreate(0, &evStart));
    CHECK_STATUS(vpiEventCreate(0, &evStop));

    // 3. Gather timings --------------------

    const int BATCH_COUNT = 20;
    const int AVERAGING_COUNT = 50;

    // Collect measurements for each execution batch
    std::vector<float> timingsMS;
    for (int batch = 0; batch < BATCH_COUNT; ++batch) {
      // Record stream queue when we start processing
      CHECK_STATUS(vpiEventRecord(evStart, stream));

      // Get the average running time within this batch.
      for (int i = 0; i < AVERAGING_COUNT; ++i) {
        // Call the algorithm to be measured.
        CHECK_STATUS(
            vpiSubmitGaussianFilter(stream, backend, image, blurred, 5, 5, 1, 1, VPI_BORDER_ZERO));
      }

      // Record stream queue just after blurring
      CHECK_STATUS(vpiEventRecord(evStop, stream));

      // Wait until the batch processing is done
      CHECK_STATUS(vpiEventSync(evStop));

      float elapsedMS;
      CHECK_STATUS(vpiEventElapsedTimeMillis(evStart, evStop, &elapsedMS));
      timingsMS.push_back(elapsedMS / AVERAGING_COUNT);
    }

    // 4. Performance analysis ----------------------

    // Get the median of the measurements so that outliers aren't considered.
    nth_element(timingsMS.begin(), timingsMS.begin() + timingsMS.size() / 2, timingsMS.end());
    float medianMS = timingsMS[timingsMS.size() / 2];

    printf("Approximated elapsed time per call: %f ms\n", medianMS);
  } catch (std::exception &e) {
    std::cerr << e.what() << std::endl;
    retval = 1;
  }

  // 4. Clean up -----------------------------------

  // Destroy stream first, it'll make sure all processing
  // submitted to it is finished.
  vpiStreamDestroy(stream);

  // Now we can destroy other VPI objects, since they aren't being
  // used anymore.
  vpiImageDestroy(image);
  vpiImageDestroy(blurred);
  vpiEventDestroy(evStart);
  vpiEventDestroy(evStop);

  return retval;
}
