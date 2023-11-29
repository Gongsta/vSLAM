#include <string>

#include "cameraparameters.hpp"
#include "imagerectifier.hpp"

const std::string DEFAULT_PARAMS_FILE = "../src/params/calibrated_params.xml";

int main(int argc, char* argv[]) {
  const cv::String keys =
      "{help h ? |           | For flags and options.           }"
      "{@settings      |" +
      DEFAULT_SETTINGS_FILE + "| input setting file            }";
  cv::CommandLineParser parser(argc, argv, keys);

  if (!parser.check()) {
    parser.printErrors();
    return 0;
  }

  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }

  //! [file_read]
  CameraParameters camera_params;
  const std::string inputSettingsFile = parser.get<std::string>(0);
  cv::FileStorage fs(inputSettingsFile, cv::FileStorage::READ);  // Read the settings
  if (!fs.isOpened()) {
    std::cout << "Could not open the configuration file: \"" << inputSettingsFile << "\""
              << std::endl;
    parser.printMessage();
    return -1;
  }

  fs["calibration_settings"] >> settings;
  fs.release();

  Calibration calibration("0", settings);
  calibration.runCalibrationAndExport();
  return 0;
}
