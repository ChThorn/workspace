#include <iostream>
#include <string>
#include <vector>
#include "workspace_definition.h"

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName 
              << " <intrinsic_file> <extrinsic_file> <marker_size_m> <output_yaml> "
              << "[visualization_path] [expected_ids (4 required)] [camera_offset_m]\n"
              << "Example: " << programName 
              << " intrinsics.yaml extrinsics.yaml 0.1 workspace.yaml viz.png 0 1 2 3 0.03\n";
}

int main(int argc, char** argv) {
    if (argc < 5 || argc > 11) {
        printUsage(argv[0]);
        return -1;
    }

    std::string intrinsic = argv[1];
    std::string extrinsic = argv[2];
    double marker_size = std::stod(argv[3]);
    std::string output = argv[4];

    std::string viz_path = "";
    std::vector<int> expected_ids;
    double camera_offset = 0.03; // Default 3cm below camera

    // Handle different argument configurations
    if (argc >= 6) {
        // Check if argument 5 is visualization path or first ID
        try {
            std::stoi(argv[5]);  // If this succeeds, it's an ID
            viz_path = "";
            for (int i = 5; i < 9 && i < argc; i++) {
                expected_ids.push_back(std::stoi(argv[i]));
            }
            // Check if camera offset is provided
            if (argc == 10) {
                camera_offset = std::stod(argv[9]);
            }
        }
        catch (...) {
            viz_path = argv[5];
            for (int i = 6; i < 10 && i < argc; i++) {
                expected_ids.push_back(std::stoi(argv[i]));
            }
            // Check if camera offset is provided
            if (argc == 11) {
                camera_offset = std::stod(argv[10]);
            }
        }
    }

    // Fill remaining IDs if not enough provided
    while (expected_ids.size() < 4) {
        expected_ids.push_back(expected_ids.size());
    }

    if (expected_ids.size() != 4) {
        std::cerr << "Exactly 4 marker IDs required" << std::endl;
        return -1;
    }

    std::cout << "Initializing workspace with camera offset: " << camera_offset << "m" << std::endl;
    
    WorkspaceDefinition workspace(intrinsic, extrinsic, marker_size, expected_ids, camera_offset);
    
    if (workspace.defineWorkspace(output, viz_path)) {
        std::cout << "Workspace definition successful!" << std::endl;
        std::cout << "Generated workspace with cuboid extending to " << camera_offset 
                  << "m below camera" << std::endl;
        return 0;
    }
    return -1;
}