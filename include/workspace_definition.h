#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/StdVector>
#include <librealsense2/rs.hpp>

class WorkspaceDefinition {
public:
    /**
     * @brief Constructor
     * @param intrinsic_file Path to camera intrinsic parameters file
     * @param extrinsic_file Path to camera-to-robot extrinsic calibration file
     * @param marker_size_meters Size of ArUco markers in meters
     * @param expected_ids Vector of expected marker IDs (4 required)
     * @param camera_offset_meters Distance below camera for workspace height (default: 0.03m)
     * @param dictionary_id ArUco dictionary ID (default: DICT_4X4_50)
     */
    WorkspaceDefinition(const std::string& intrinsic_file, 
                        const std::string& extrinsic_file,
                        double marker_size_meters,
                        const std::vector<int>& expected_ids,
                        double camera_offset_meters = 0.03,
                        double height_above_markers = 0.6,
                        double height_below_markers = 0.03,
                        bool use_camera_intrinsics = false,  // New parameter
                        int dictionary_id = cv::aruco::DICT_4X4_50);
    
    ~WorkspaceDefinition();
    
    bool initializeCamera();
    cv::Mat captureFrame();
    int detectMarkers(const cv::Mat& image);
    std::vector<Eigen::Vector3d> calculateWorkspaceCorners();
    bool saveWorkspaceToYAML(const std::string& filename);
    bool defineWorkspace(const std::string& output_yaml, 
                         const std::string& visualization_path = "");
    cv::Mat getMarkerVisualization();
    bool getCameraIntrinsics();

private:
    // Configuration
    const std::vector<int> expected_marker_ids;
    
    // Camera parameters
    rs2::pipeline pipe;
    rs2::config cfg;
    const int CAMERA_WIDTH = 640;
    const int CAMERA_HEIGHT = 480;
    bool camera_initialized = false;

    // ArUco parameters
    cv::aruco::Dictionary dictionary;
    cv::aruco::DetectorParameters parameters;
    double marker_size;
    double camera_offset_meters; // Distance below camera for workspace height

    // Calibration data
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    Eigen::Matrix4d bHc;

    // Detection results
    std::vector<int> marker_ids;
    std::vector<std::vector<cv::Point2f>> marker_corners;
    std::vector<cv::Vec3d> rvecs, tvecs;

    // Workspace data
    std::vector<Eigen::Vector3d> workspace_corners;
    double x_min, x_max, y_min, y_max, z_min, z_max;

    // Helper methods
    bool loadCalibrationData(const std::string& intrinsic_file, const std::string& extrinsic_file);
    void calculateWorkspaceBoundaries();
    Eigen::Matrix4d cameraToRobotTransform(const cv::Vec3d& rvec, const cv::Vec3d& tvec);
    void sortMarkersById();
    Eigen::Vector3d camera_position;
    
    double height_above_markers; // Height to extend above markers
    double height_below_markers; // Height to extend below markers

    // Optional: add storage for per-marker ceiling and floor points
    std::vector<Eigen::Vector3d> ceiling_points;
    std::vector<Eigen::Vector3d> floor_points;

    bool use_camera_intrinsics;  // Flag to use camera intrinsics instead of file
};