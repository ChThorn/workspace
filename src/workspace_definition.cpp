#include "workspace_definition.h"
#include <iostream>
#include <limits>
#include <algorithm>
#include <opencv2/core/eigen.hpp>
#include <fstream>

#define PI 3.1415926
const double SAFETY_MARGIN = 0.01;  // 1cm margin

WorkspaceDefinition::WorkspaceDefinition(const std::string& intrinsic_file, 
                                       const std::string& extrinsic_file,
                                       double marker_size_meters,
                                       const std::vector<int>& expected_ids,
                                       double camera_offset_meters,
                                       double height_above_markers,
                                       double height_below_markers,
                                       bool use_camera_intrinsics,  // New parameter
                                       int dictionary_id)
    : expected_marker_ids(expected_ids),
      marker_size(marker_size_meters),
      camera_offset_meters(camera_offset_meters),
      height_above_markers(height_above_markers),
      height_below_markers(height_below_markers),
      use_camera_intrinsics(use_camera_intrinsics),  // Initialize the new flag
      x_min(std::numeric_limits<double>::max()),
      x_max(std::numeric_limits<double>::lowest()),
      y_min(std::numeric_limits<double>::max()),
      y_max(std::numeric_limits<double>::lowest()),
      z_min(std::numeric_limits<double>::max()),
      z_max(std::numeric_limits<double>::lowest())
{
    if(height_above_markers <= 0 || height_below_markers <= 0) {
        throw std::invalid_argument("Height parameters must be positive");
    }
    dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::PredefinedDictionaryType(dictionary_id));
    parameters = cv::aruco::DetectorParameters();

    // Initialize the camera early if we need to use its intrinsics
    if (use_camera_intrinsics) {
        if (!initializeCamera()) {
            throw std::runtime_error("Failed to initialize camera to get intrinsics");
        }
    }
    
    loadCalibrationData(intrinsic_file, extrinsic_file);
}

WorkspaceDefinition::~WorkspaceDefinition() {
    if (camera_initialized) {
        try {
            pipe.stop(); // Safe to call even if already stopped
        } catch (const rs2::error& e) {
            std::cerr << "Pipeline stop error: " << e.what() << std::endl;
        }
    }
}

bool WorkspaceDefinition::loadCalibrationData(const std::string& intrinsic_file, 
                                            const std::string& extrinsic_file) 
{
    // Check if we should use camera intrinsics or load from file
    if (use_camera_intrinsics) {
        std::cout << "Using built-in camera intrinsics instead of file" << std::endl;
        if (!getCameraIntrinsics()) {
            std::cerr << "Failed to get intrinsics from camera" << std::endl;
            return false;
        }
    } else {
        // Check if the intrinsic file exists
        std::ifstream file_check(intrinsic_file.c_str());
        if (!file_check.good()) {
            std::cout << "Intrinsic file not found: " << intrinsic_file 
                      << ", trying to use camera intrinsics" << std::endl;
            
            // Initialize camera if not already done
            if (!camera_initialized && !initializeCamera()) {
                std::cerr << "Failed to initialize camera for intrinsics" << std::endl;
                return false;
            }
            
            if (!getCameraIntrinsics()) {
                std::cerr << "Failed to get intrinsics from camera" << std::endl;
                return false;
            }
        } else {
            // File exists, load as normal
            file_check.close();
            
            cv::FileStorage fs_intrinsic(intrinsic_file, cv::FileStorage::READ);
            if (!fs_intrinsic.isOpened()) {
                std::cerr << "Failed to open intrinsic file: " << intrinsic_file << std::endl;
                return false;
            }
            fs_intrinsic["camera_matrix"] >> camera_matrix;
            fs_intrinsic["distortion_coefficients"] >> dist_coeffs;
            fs_intrinsic.release();
            
            std::cout << "Loaded intrinsics from file: " << intrinsic_file << std::endl;
        }
    }

    // Rest of the method remains the same (loading extrinsics)
    cv::FileStorage fs_extrinsic(extrinsic_file, cv::FileStorage::READ);
    if (!fs_extrinsic.isOpened()) {
        std::cerr << "Failed to open extrinsic file: " << extrinsic_file << std::endl;
        return false;
    }
    
    cv::Mat bHc_cv;
    fs_extrinsic["bHc"] >> bHc_cv;
    fs_extrinsic.release();

    if (bHc_cv.rows != 4 || bHc_cv.cols != 4) {
        std::cerr << "Invalid extrinsic matrix dimensions (must be 4x4)" << std::endl;
        return false;
    }
    cv::cv2eigen(bHc_cv, bHc);

    // Compute camera position: bHc * [0,0,0,1] (camera origin in robot frame)
    Eigen::Vector4d cam_pos_homog = bHc * Eigen::Vector4d(0, 0, 0, 1);
    camera_position = cam_pos_homog.head<3>();
    
    std::cout << "Camera position in robot frame: [" 
              << camera_position.x() << ", " 
              << camera_position.y() << ", " 
              << camera_position.z() << "]" << std::endl;
              
    return true;
}

bool WorkspaceDefinition::initializeCamera() {
    try {
        cfg.enable_stream(RS2_STREAM_COLOR, CAMERA_WIDTH, CAMERA_HEIGHT, RS2_FORMAT_BGR8, 30);
        pipe.start(cfg);
        
        // Get the active profile and cast to video stream profile
        auto profile = pipe.get_active_profile();
        auto video_profile = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
        
        // Validate resolution
        if (video_profile.width() != CAMERA_WIDTH || video_profile.height() != CAMERA_HEIGHT) {
            std::cerr << "Unsupported camera resolution. Got " 
                      << video_profile.width() << "x" << video_profile.height()
                      << ", expected " << CAMERA_WIDTH << "x" << CAMERA_HEIGHT << std::endl;
            return false;
        }

        for (int i = 0; i < 30; i++) pipe.wait_for_frames();
        camera_initialized = true;
        return true;
    } 
    catch (const rs2::error& e) {
        std::cerr << "RealSense error: " << e.what() << std::endl;
        return false;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return false;
    }
}

cv::Mat WorkspaceDefinition::captureFrame() {
    if (!camera_initialized) return cv::Mat();
    
    try {
        auto frames = pipe.wait_for_frames();
        auto color_frame = frames.get_color_frame();
        return cv::Mat(cv::Size(CAMERA_WIDTH, CAMERA_HEIGHT), CV_8UC3, 
            (void*)color_frame.get_data(), cv::Mat::AUTO_STEP).clone();
    }
    catch (const rs2::error& e) {
        std::cerr << "Frame capture error: " << e.what() << std::endl;
        return cv::Mat();
    }
}

int WorkspaceDefinition::detectMarkers(const cv::Mat& image) {
    if (image.empty()) return 0;

    marker_ids.clear();
    marker_corners.clear();
    rvecs.clear();
    tvecs.clear();

    cv::aruco::ArucoDetector detector(dictionary, parameters);
    detector.detectMarkers(image, marker_corners, marker_ids);

    // Filter markers by expected IDs
    std::vector<int> filtered_ids;
    std::vector<std::vector<cv::Point2f>> filtered_corners;
    for (size_t i = 0; i < marker_ids.size(); i++) {
        if (std::find(expected_marker_ids.begin(), expected_marker_ids.end(), marker_ids[i]) 
            != expected_marker_ids.end()) {
            filtered_ids.push_back(marker_ids[i]);
            filtered_corners.push_back(marker_corners[i]);
        }
    }
    marker_ids = filtered_ids;
    marker_corners = filtered_corners;

    if (marker_ids.empty()) return 0;

    // Estimate poses
    float halfSize = marker_size / 2.0f;
    std::vector<cv::Point3f> objPoints = {
        {-halfSize, halfSize, 0}, {halfSize, halfSize, 0},
        {halfSize, -halfSize, 0}, {-halfSize, -halfSize, 0}
    };

    rvecs.resize(marker_ids.size());
    tvecs.resize(marker_ids.size());
    
    for (size_t i = 0; i < marker_ids.size(); i++) {
        if (!cv::solvePnP(objPoints, marker_corners[i], camera_matrix, 
                         dist_coeffs, rvecs[i], tvecs[i])) {
            std::cerr << "Pose estimation failed for marker " << marker_ids[i] << std::endl;
            return 0;
        }
    }
    
    return marker_ids.size();
}

void WorkspaceDefinition::sortMarkersById() {
    std::vector<std::pair<int, size_t>> id_index_pairs;
    for (size_t i = 0; i < marker_ids.size(); i++)
        id_index_pairs.emplace_back(marker_ids[i], i);
    
    std::sort(id_index_pairs.begin(), id_index_pairs.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<int> sorted_ids;
    std::vector<std::vector<cv::Point2f>> sorted_corners;
    std::vector<cv::Vec3d> sorted_rvecs, sorted_tvecs;

    for (const auto& pair : id_index_pairs) {
        sorted_ids.push_back(marker_ids[pair.second]);
        sorted_corners.push_back(marker_corners[pair.second]);
        sorted_rvecs.push_back(rvecs[pair.second]);
        sorted_tvecs.push_back(tvecs[pair.second]);
    }

    marker_ids = sorted_ids;
    marker_corners = sorted_corners;
    rvecs = sorted_rvecs;
    tvecs = sorted_tvecs;
}

std::vector<Eigen::Vector3d> WorkspaceDefinition::calculateWorkspaceCorners() {
    workspace_corners.clear();
    ceiling_points.clear();  // Clear these vectors
    floor_points.clear();
    
    if (marker_ids.empty()) return workspace_corners;

    sortMarkersById();

    for (size_t i = 0; i < marker_ids.size(); i++) {
        Eigen::Matrix4d bHm = cameraToRobotTransform(rvecs[i], tvecs[i]);
        workspace_corners.emplace_back(bHm(0, 3), bHm(1, 3), bHm(2, 3));
        
        std::cout << "Marker " << marker_ids[i] << " position: [" 
                  << bHm(0, 3) << ", " << bHm(1, 3) << ", " << bHm(2, 3) << "]" << std::endl;
    }

    calculateWorkspaceBoundaries();
    
    // After calculating boundaries, populate ceiling and floor points
    for (const auto& corner : workspace_corners) {
        // Create points at the same x,y but at ceiling (z_max) and floor (z_min) heights
        ceiling_points.push_back(Eigen::Vector3d(corner.x(), corner.y(), z_max));
        floor_points.push_back(Eigen::Vector3d(corner.x(), corner.y(), z_min));
    }
    
    return workspace_corners;
}

void WorkspaceDefinition::calculateWorkspaceBoundaries() {
    // Calculate base Z from marker positions
    double base_z = 0.0;
    for(const auto& corner : workspace_corners) {
        base_z += corner.z();
    }
    base_z /= workspace_corners.size();

    // Apply height parameters relative to markers
    z_max = base_z + height_above_markers;
    z_min = base_z - height_below_markers;

    // X-Y boundaries from markers
    x_min = y_min = std::numeric_limits<double>::max();
    x_max = y_max = std::numeric_limits<double>::lowest();
    
    for(const auto& corner : workspace_corners) {
        x_min = std::min(x_min, corner.x());
        x_max = std::max(x_max, corner.x());
        y_min = std::min(y_min, corner.y());
        y_max = std::max(y_max, corner.y());
    }

    // Apply safety margins
    x_min -= SAFETY_MARGIN; x_max += SAFETY_MARGIN;
    y_min -= SAFETY_MARGIN; y_max += SAFETY_MARGIN;
    z_min -= SAFETY_MARGIN; z_max += SAFETY_MARGIN;
}

bool WorkspaceDefinition::saveWorkspaceToYAML(const std::string& filename) {
    if (workspace_corners.size() != 4) {
        std::cerr << "Error: Expected 4 workspace corners, got " 
                  << workspace_corners.size() << std::endl;
        return false;
    }

    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return false;
    }

    fs << "workspace_boundaries" << "{"
       << "x_min" << x_min << "x_max" << x_max
       << "y_min" << y_min << "y_max" << y_max
       << "z_min" << z_min << "z_max" << z_max << "}";

    fs << "height_parameters" << "{"
       << "height_above_markers" << height_above_markers
       << "height_below_markers" << height_below_markers << "}";

    fs << "markers" << "[";
    for (size_t i = 0; i < 4; i++) {
        fs << "{:" << "id" << marker_ids[i] 
           << "position" << "[:"
           << workspace_corners[i].x() << workspace_corners[i].y() 
           << workspace_corners[i].z() << "]"
           << "ceiling" << "[:"
           << ceiling_points[i].x() << ceiling_points[i].y() 
           << ceiling_points[i].z() << "]"
           << "floor" << "[:"
           << floor_points[i].x() << floor_points[i].y() 
           << floor_points[i].z() << "]"
           << "}";
    }
    fs << "]";
    
    fs.release();
    std::cout << "Workspace configuration saved to: " << filename << std::endl;
    return true;
}

bool WorkspaceDefinition::defineWorkspace(const std::string& output_yaml, 
                                        const std::string& visualization_path) 
{
    if (!camera_initialized && !initializeCamera()) {
        std::cerr << "Failed to initialize camera" << std::endl;
        return false;
    }

    cv::Mat image = captureFrame();
    if (image.empty()) {
        std::cerr << "Failed to capture frame" << std::endl;
        return false;
    }

    int num_markers = detectMarkers(image);
    if (num_markers != 4) {
        std::cerr << "Exactly 4 markers required. Detected: " << num_markers << std::endl;
        return false;
    }

    calculateWorkspaceCorners();
    if (workspace_corners.size() != 4) {
        std::cerr << "Failed to calculate workspace corners" << std::endl;
        return false;
    }

    if (!visualization_path.empty()) {
        cv::Mat vis = getMarkerVisualization();
        if (!vis.empty()) {
            cv::imwrite(visualization_path, vis);
            std::cout << "Visualization saved to: " << visualization_path << std::endl;
        } else {
            std::cerr << "Failed to create visualization" << std::endl;
        }
    }

    return saveWorkspaceToYAML(output_yaml);
}

cv::Mat WorkspaceDefinition::getMarkerVisualization() {
    // First capture the image
    cv::Mat image = captureFrame();
    if (image.empty()) return cv::Mat();

    // Draw detected markers
    cv::aruco::drawDetectedMarkers(image, marker_corners, marker_ids);

    // Instead of trying to project 3D workspace boundaries, we'll draw a simplified
    // visualization that connects the detected marker corners directly in the image
    if (marker_corners.size() == 4) {
        // Get corner center points (average of the 4 corners of each marker)
        std::vector<cv::Point2f> marker_centers;
        for (const auto& corners : marker_corners) {
            cv::Point2f center(0, 0);
            for (const auto& corner : corners) {
                center += corner;
            }
            center *= 0.25f; // Average of 4 points
            marker_centers.push_back(center);
        }
        
        // Sort centers to match the order of workspace_corners
        if (marker_ids.size() == marker_centers.size()) {
            std::vector<cv::Point2f> sorted_centers(marker_centers.size());
            for (size_t i = 0; i < marker_ids.size(); i++) {
                // Find the index of this ID in the workspace_corners
                auto it = std::find(expected_marker_ids.begin(), expected_marker_ids.end(), marker_ids[i]);
                if (it != expected_marker_ids.end()) {
                    size_t idx = std::distance(expected_marker_ids.begin(), it);
                    if (idx < sorted_centers.size()) {
                        sorted_centers[idx] = marker_centers[i];
                    }
                }
            }
            marker_centers = sorted_centers;
        }

        // Connect marker centers to form the bottom face of the workspace
        for (size_t i = 0; i < marker_centers.size(); i++) {
            cv::line(image, marker_centers[i], marker_centers[(i+1) % marker_centers.size()], 
                    cv::Scalar(0, 0, 255), 2);
        }

        // Add text labels for clarity
        std::string offset_text = "Camera offset: " + 
                                 std::to_string(camera_offset_meters*100) + "cm";
        cv::putText(image, "Red: Marker boundaries", cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
        cv::putText(image, offset_text, cv::Point(10, 60), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);
        
        // Add workspace dimensions
        std::string dim_text = "Workspace dimensions: " + 
                              std::to_string(std::round((x_max - x_min) * 100)/100) + "x" + 
                              std::to_string(std::round((y_max - y_min) * 100)/100) + "x" + 
                              std::to_string(std::round((z_max - z_min) * 100)/100) + "m";
        cv::putText(image, dim_text, cv::Point(10, 90), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
    }

    return image;
}

Eigen::Matrix4d WorkspaceDefinition::cameraToRobotTransform(const cv::Vec3d& rvec, 
                                                           const cv::Vec3d& tvec) 
{
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    
    Eigen::Matrix4d cHm = Eigen::Matrix4d::Identity();
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            cHm(r, c) = R.at<double>(r, c);
        }
    }
    cHm(0, 3) = tvec[0];
    cHm(1, 3) = tvec[1];
    cHm(2, 3) = tvec[2];
    
    return bHc * cHm;
}

bool WorkspaceDefinition::getCameraIntrinsics() {
    if (!camera_initialized) {
        std::cerr << "Camera not initialized, cannot get intrinsics" << std::endl;
        return false;
    }
    
    try {
        // Get the active profile and extract intrinsics
        auto profile = pipe.get_active_profile();
        auto video_profile = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
        rs2_intrinsics intrinsics = video_profile.get_intrinsics();
        
        // Convert to OpenCV format
        camera_matrix = cv::Mat::eye(3, 3, CV_64F);
        camera_matrix.at<double>(0, 0) = intrinsics.fx;
        camera_matrix.at<double>(0, 2) = intrinsics.ppx;
        camera_matrix.at<double>(1, 1) = intrinsics.fy;
        camera_matrix.at<double>(1, 2) = intrinsics.ppy;
        
        // Distortion coefficients
        dist_coeffs = cv::Mat::zeros(1, 5, CV_64F);
        for (int i = 0; i < 5; i++) {
            dist_coeffs.at<double>(0, i) = intrinsics.coeffs[i];
        }
        
        std::cout << "Using intrinsics from camera:" << std::endl;
        std::cout << "  fx: " << intrinsics.fx << ", fy: " << intrinsics.fy << std::endl;
        std::cout << "  ppx: " << intrinsics.ppx << ", ppy: " << intrinsics.ppy << std::endl;
        std::cout << "  Distortion model: " << intrinsics.model << std::endl;
        
        return true;
    }
    catch (const rs2::error& e) {
        std::cerr << "RealSense error getting intrinsics: " << e.what() << std::endl;
        return false;
    }
    catch (const std::exception& e) {
        std::cerr << "Error getting intrinsics: " << e.what() << std::endl;
        return false;
    }
}