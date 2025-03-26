// #include <iostream>
// #include <string>
// #include <vector>
// #include "workspace_definition.h"

// void printUsage(const char* programName) {
//     std::cout << "Usage: " << programName 
//               << " <intrinsic_file> <extrinsic_file> <marker_size_m> <output_yaml> "
//               << "[visualization_path] [expected_ids (4 required)] [camera_offset_m] [height_above] [height_below]\n"
//               << "Example: " << programName 
//               << " intrinsics.yaml extrinsics.yaml 0.1 workspace.yaml viz.png 0 1 2 3 0.03 0.6 0.03\n";
// }

// bool parseArguments(int argc, char** argv, 
//                     std::string& viz_path,
//                     std::vector<int>& expected_ids,
//                     double& camera_offset,
//                     double& height_above,
//                     double& height_below) 
// {
//     height_above = 0.6;  // Default values
//     height_below = 0.03;
//     camera_offset = 0.03;

//     // Case 1: Visualization path provided
//     if (argc >= 6 && !std::isdigit(argv[5][0])) {
//         viz_path = argv[5];
        
//         // Parse IDs (positions 6-9)
//         for(int i=6; i<10 && i<argc; i++) {
//             try {
//                 expected_ids.push_back(std::stoi(argv[i]));
//             } catch (...) {
//                 std::cerr << "Invalid ID format: " << argv[i] << std::endl;
//                 return false;
//             }
//         }

//         // Parse optional parameters starting from position 10
//         if(argc >= 11) camera_offset = std::stod(argv[10]);
//         if(argc >= 12) height_above = std::stod(argv[11]);
//         if(argc >= 13) height_below = std::stod(argv[12]);
//     }
//     // Case 2: No visualization path
//     else if (argc >= 5) {
//         // Parse IDs (positions 5-8)
//         for(int i=5; i<9 && i<argc; i++) {
//             try {
//                 expected_ids.push_back(std::stoi(argv[i]));
//             } catch (...) {
//                 std::cerr << "Invalid ID format: " << argv[i] << std::endl;
//                 return false;
//             }
//         }

//         // Parse optional parameters starting from position 9
//         if(argc >= 10) camera_offset = std::stod(argv[9]);
//         if(argc >= 11) height_above = std::stod(argv[10]);
//         if(argc >= 12) height_below = std::stod(argv[11]);
//     }

//     // Fill missing IDs with placeholders
//     if (expected_ids.size() != 4) {
//         std::cerr << "Exactly 4 valid marker IDs required!" << std::endl;
//         return false;
//     }

//     if(expected_ids.size() != 4) {
//         std::cerr << "Exactly 4 marker IDs required" << std::endl;
//         return false;
//     }

//     return true;
// }

// int main(int argc, char** argv) {
//     if (argc < 5 || argc > 13) {
//         printUsage(argv[0]);
//         return -1;
//     }

//     // Required parameters
//     std::string intrinsic = argv[1];
//     std::string extrinsic = argv[2];
//     double marker_size = std::stod(argv[3]);
//     std::string output = argv[4];

//     // Optional parameters
//     std::string viz_path;
//     std::vector<int> expected_ids;
//     double camera_offset, height_above, height_below;

//     if(!parseArguments(argc, argv, viz_path, expected_ids, 
//                       camera_offset, height_above, height_below)) {
//         return -1;
//     }

//     WorkspaceDefinition workspace(intrinsic, extrinsic, marker_size, expected_ids,
//                                   camera_offset, height_above, height_below);

//     std::cout << "Initializing workspace with parameters:\n"
//               << "  Camera offset: " << camera_offset << "m\n"
//               << "  Height above markers: " << height_above << "m\n"
//               << "  Height below markers: " << height_below << "m\n";

//     if (workspace.defineWorkspace(output, viz_path)) {
//         std::cout << "Workspace definition successful!\n";
//         return 0;
//     }
//     return -1;
// }

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sys/stat.h>
#include <opencv2/opencv.hpp>
#include "workspace_definition.h"
#include <yaml-cpp/yaml.h>

// Default configuration file path (in current directory)
const std::string DEFAULT_CONFIG_PATH = "../data/workspace_params_config.yaml";

struct WorkspaceConfig {
    std::string intrinsic_file = "../data/intrinsic_params.yml";
    std::string extrinsic_file = "../data/test_extrinsic.yaml";
    double marker_size = 0.05;
    std::string output_yaml = "../data/workspace.yaml";
    std::string visualization_path = "../data/visual.png";
    std::vector<int> expected_ids = {0, 1, 2, 3};
    double camera_offset = 0.03;
    double height_above = 0.6;
    double height_below = 0.03;
};

// Helper function to check if directory exists
bool directoryExists(const std::string& path) {
    struct stat info;
    return stat(path.c_str(), &info) == 0 && (info.st_mode & S_IFDIR);
}

// Helper function to check if file exists
bool fileExists(const std::string& filename) {
    std::ifstream f(filename.c_str());
    return f.good();
}

// Helper function to ensure directory exists
bool ensureDirectoryExists(const std::string& path) {
    size_t pos = 0;
    std::string dir;
    while ((pos = path.find('/', pos + 1)) != std::string::npos) {
        dir = path.substr(0, pos);
        if (dir.empty()) continue; // Skip root directory
        
        if (!directoryExists(dir)) {
            #ifdef _WIN32
            if (mkdir(dir.c_str()) != 0) {
            #else
            if (mkdir(dir.c_str(), 0755) != 0) {
            #endif
                std::cerr << "Failed to create directory: " << dir << std::endl;
                return false;
            }
        }
    }
    return true;
}

void printUsage(const char* programName) {
    std::cout << "Usage options:\n"
              << "1. " << programName << "                  (Uses default config file: " << DEFAULT_CONFIG_PATH << ")\n"
              << "2. " << programName << " config.yaml      (Uses specified config file)\n"
              << "3. " << programName << " --no-config <intrinsic_file> <extrinsic_file> <marker_size_m> <output_yaml> "
              << "[visualization_path] [expected_ids (4 required)] [camera_offset_m] [height_above] [height_below]\n"
              << "Example: " << programName 
              << " --no-config intrinsics.yaml extrinsics.yaml 0.1 workspace.yaml viz.png 0 1 2 3 0.03 0.6 0.03\n";
}

// bool loadConfigFile(const std::string& filename, WorkspaceConfig& config) {
//     // Check if the file exists first
//     if (!fileExists(filename)) {
//         std::cerr << "Config file does not exist: " << filename << std::endl;
//         return false;
//     }

//     try {
//         // Manual parsing only - skip OpenCV FileStorage entirely
//         std::ifstream inFile(filename);
//         if (!inFile.is_open()) {
//             std::cerr << "Failed to open config file: " << filename << std::endl;
//             return false;
//         }
        
//         std::string line;
//         while (std::getline(inFile, line)) {
//             // Trim whitespace
//             line.erase(0, line.find_first_not_of(" \t"));
            
//             // Skip empty lines and comments
//             if (line.empty() || line[0] == '#')
//                 continue;
            
//             // Parse key-value pairs
//             size_t colonPos = line.find(':');
//             if (colonPos != std::string::npos) {
//                 std::string key = line.substr(0, colonPos);
//                 std::string value = line.substr(colonPos + 1);
                
//                 // Trim key and value
//                 key.erase(0, key.find_first_not_of(" \t"));
//                 key.erase(key.find_last_not_of(" \t") + 1);
//                 value.erase(0, value.find_first_not_of(" \t"));
//                 value.erase(value.find_last_not_of(" \t") + 1);
                
//                 // Remove quotes from string values
//                 if (value.size() >= 2 && value[0] == '"' && value[value.size()-1] == '"') {
//                     value = value.substr(1, value.size() - 2);
//                 }
                
//                 // Parse marker_ids array
//                 if (key == "marker_ids") {
//                     config.expected_ids.clear();
//                     size_t startBracket = value.find('[');
//                     size_t endBracket = value.find(']');
//                     if (startBracket != std::string::npos && endBracket != std::string::npos) {
//                         std::string idsStr = value.substr(startBracket + 1, endBracket - startBracket - 1);
//                         std::istringstream iss(idsStr);
//                         std::string token;
//                         while (std::getline(iss, token, ',')) {
//                             token.erase(0, token.find_first_not_of(" \t"));
//                             token.erase(token.find_last_not_of(" \t") + 1);
//                             if (!token.empty()) {
//                                 config.expected_ids.push_back(std::stoi(token));
//                             }
//                         }
//                     }
//                     continue;
//                 }
                
//                 // Parse other values
//                 if (key == "intrinsic_file") {
//                     config.intrinsic_file = value;
//                 } else if (key == "extrinsic_file") {
//                     config.extrinsic_file = value;
//                 } else if (key == "output_yaml") {
//                     config.output_yaml = value;
//                 } else if (key == "visualization_path") {
//                     config.visualization_path = value;
//                 } else if (key == "marker_size") {
//                     config.marker_size = std::stod(value);
//                 } else if (key == "camera_offset") {
//                     config.camera_offset = std::stod(value);
//                 } else if (key == "height_above") {
//                     config.height_above = std::stod(value);
//                 } else if (key == "height_below") {
//                     config.height_below = std::stod(value);
//                 }
//             }
//         }
        
//         inFile.close();
        
//         // Validate marker IDs
//         if (config.expected_ids.size() != 4) {
//             std::cerr << "Config error: Exactly 4 marker IDs required" << std::endl;
//             return false;
//         }

//         return true;
//     } catch (const std::exception& e) {
//         std::cerr << "Error while loading config: " << e.what() << std::endl;
//         return false;
//     }
// }

bool loadConfigFile(const std::string& filename, WorkspaceConfig& config) {
    try {
        YAML::Node configFile = YAML::LoadFile(filename);
        
        config.intrinsic_file = configFile["intrinsic_file"].as<std::string>();
        config.extrinsic_file = configFile["extrinsic_file"].as<std::string>();
        config.marker_size = configFile["marker_size"].as<double>();
        config.output_yaml = configFile["output_yaml"].as<std::string>();
        config.visualization_path = configFile["visualization_path"].as<std::string>();
        
        // Parse marker IDs
        config.expected_ids.clear();
        YAML::Node ids = configFile["marker_ids"];
        for(auto&& id : ids) {
            config.expected_ids.push_back(id.as<int>());
        }
        
        config.camera_offset = configFile["camera_offset"].as<double>();
        config.height_above = configFile["height_above"].as<double>();
        config.height_below = configFile["height_below"].as<double>();

        // Validate marker IDs
        if(config.expected_ids.size() != 4) {
            throw std::runtime_error("Exactly 4 marker IDs required");
        }
        
        return true;
    } catch(const YAML::Exception& e) {
        std::cerr << "YAML Error: " << e.what() << std::endl;
        return false;
    } catch(const std::exception& e) {
        std::cerr << "Config Error: " << e.what() << std::endl;
        return false;
    }
}


bool saveDefaultConfig() {
    try {
        WorkspaceConfig defaultConfig;
        
        // Use standard C++ file I/O instead of OpenCV
        std::ofstream configFile(DEFAULT_CONFIG_PATH);
        if (!configFile.is_open()) {
            std::cerr << "Failed to create default config file: " << DEFAULT_CONFIG_PATH << std::endl;
            return false;
        }
        
        // Write YAML content directly
        configFile << "# Workspace Tool Configuration\n\n";
        configFile << "# File paths\n";
        configFile << "intrinsic_file: \"" << defaultConfig.intrinsic_file << "\"\n";
        configFile << "extrinsic_file: \"" << defaultConfig.extrinsic_file << "\"\n";
        configFile << "output_yaml: \"" << defaultConfig.output_yaml << "\"\n";
        configFile << "visualization_path: \"" << defaultConfig.visualization_path << "\"\n\n";
        
        configFile << "# Marker parameters\n";
        configFile << "marker_size: " << defaultConfig.marker_size << "  # in meters\n";
        configFile << "marker_ids: [";
        for (size_t i = 0; i < defaultConfig.expected_ids.size(); i++) {
            configFile << defaultConfig.expected_ids[i];
            if (i < defaultConfig.expected_ids.size() - 1) {
                configFile << ", ";
            }
        }
        configFile << "]\n\n";
        
        configFile << "# Workspace parameters\n";
        configFile << "camera_offset: " << defaultConfig.camera_offset << "  # in meters\n";
        configFile << "height_above: " << defaultConfig.height_above << "  # in meters\n";
        configFile << "height_below: " << defaultConfig.height_below << "  # in meters\n";
        
        configFile.close();
        
        std::cout << "Created default config file: " << DEFAULT_CONFIG_PATH << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error while saving config: " << e.what() << std::endl;
        return false;
    }
}

bool parseArguments(int argc, char** argv, 
                    std::string& config_file,
                    bool& use_config_file,
                    WorkspaceConfig& config) 
{
    // No arguments - use default config or create one
    if (argc == 1) {
        config_file = DEFAULT_CONFIG_PATH;
        
        // Check if default config exists
        if (fileExists(DEFAULT_CONFIG_PATH)) {
            use_config_file = true;
            return true;
        } else {
            // Create default config but don't try to read it back immediately
            std::cout << "Default config not found. Creating one..." << std::endl;
            if (saveDefaultConfig()) {
                std::cout << "Created default config file for future use." << std::endl;
            } else {
                std::cerr << "Failed to create default config file." << std::endl;
            }
            
            // Since we just created the default config with known values,
            // we'll use the hardcoded defaults directly instead of reading the file
            use_config_file = false;
            return true;
        }
    }
    
    // One argument - either it's a config file or --help
    if (argc == 2) {
        std::string arg = argv[1];
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return false;
        }
        else if (arg == "--no-config") {
            std::cerr << "Error: --no-config requires additional parameters" << std::endl;
            printUsage(argv[0]);
            return false;
        }
        else {
            // Assume it's a config file
            config_file = arg;
            use_config_file = true;
            
            // Verify file exists
            if (!fileExists(config_file)) {
                std::cerr << "Config file not found: " << config_file << std::endl;
                return false;
            }
            return true;
        }
    }
    
    // Multiple arguments - check if using --no-config
    if (std::string(argv[1]) == "--no-config") {
        use_config_file = false;
        
        if (argc < 6) {
            std::cerr << "Error: Not enough parameters for --no-config mode" << std::endl;
            printUsage(argv[0]);
            return false;
        }
        
        // Parse required parameters
        config.intrinsic_file = argv[2];
        config.extrinsic_file = argv[3];
        try {
            config.marker_size = std::stod(argv[4]);
        } catch (const std::exception& e) {
            std::cerr << "Invalid marker size: " << argv[4] << std::endl;
            return false;
        }
        config.output_yaml = argv[5];
        
        // Optional parameters
        if (argc >= 7) config.visualization_path = argv[6];
        
        // Parse marker IDs
        config.expected_ids.clear();
        for(int i = 7; i < 11 && i < argc; i++) {
            try {
                config.expected_ids.push_back(std::stoi(argv[i]));
            } catch (...) {
                std::cerr << "Invalid ID format: " << argv[i] << std::endl;
                return false;
            }
        }
        
        // Fill missing IDs with defaults
        while(config.expected_ids.size() < 4) {
            config.expected_ids.push_back(config.expected_ids.size());
        }
        
        // Parse additional parameters
        if(argc >= 12) {
            try {
                config.camera_offset = std::stod(argv[11]);
            } catch (...) {
                std::cerr << "Invalid camera offset: " << argv[11] << std::endl;
                return false;
            }
        }
        if(argc >= 13) {
            try {
                config.height_above = std::stod(argv[12]);
            } catch (...) {
                std::cerr << "Invalid height above value: " << argv[12] << std::endl;
                return false;
            }
        }
        if(argc >= 14) {
            try {
                config.height_below = std::stod(argv[13]);
            } catch (...) {
                std::cerr << "Invalid height below value: " << argv[13] << std::endl;
                return false;
            }
        }
        
        return true;
    }
    
    // Invalid argument format
    std::cerr << "Error: Invalid arguments" << std::endl;
    printUsage(argv[0]);
    return false;
}

int main(int argc, char** argv) {
    try {
        WorkspaceConfig config;
        
        // Simple argument handling to avoid config file errors
        if (argc == 1) {
            // No arguments - just use default values directly
            // Create a config file for future use, but don't try to read it now
            if (!fileExists(DEFAULT_CONFIG_PATH)) {
                saveDefaultConfig();
                std::cout << "Created config file for future use: " << DEFAULT_CONFIG_PATH << std::endl;
            } else {
                // File exists, try to load it
                if (loadConfigFile(DEFAULT_CONFIG_PATH, config)) {
                    std::cout << "Using configuration from: " << DEFAULT_CONFIG_PATH << std::endl;
                }
            }
        } else if (argc == 2) {
            // One argument - could be a config file
            std::string arg = argv[1];
            if (arg == "--help" || arg == "-h") {
                printUsage(argv[0]);
                return 0;
            } else if (fileExists(arg)) {
                // Try to load specified config file
                if (loadConfigFile(arg, config)) {
                    std::cout << "Using configuration from: " << arg << std::endl;
                } else {
                    std::cerr << "Failed to parse config file, using defaults." << std::endl;
                }
            } else {
                std::cerr << "Config file not found: " << arg << std::endl;
                printUsage(argv[0]);
                return -1;
            }
        } else if (std::string(argv[1]) == "--no-config" && argc >= 6) {
            // Command line parameters without config
            config.intrinsic_file = argv[2];
            config.extrinsic_file = argv[3];
            config.marker_size = std::stod(argv[4]);
            config.output_yaml = argv[5];
            
            if (argc >= 7) config.visualization_path = argv[6];
            
            // Parse marker IDs
            config.expected_ids.clear();
            for(int i = 7; i < 11 && i < argc; i++) {
                config.expected_ids.push_back(std::stoi(argv[i]));
            }
            
            // Fill missing IDs with defaults
            while(config.expected_ids.size() < 4) {
                config.expected_ids.push_back(config.expected_ids.size());
            }
            
            // Parse additional parameters
            if(argc >= 12) config.camera_offset = std::stod(argv[11]);
            if(argc >= 13) config.height_above = std::stod(argv[12]);
            if(argc >= 14) config.height_below = std::stod(argv[13]);
        } else {
            printUsage(argv[0]);
            return -1;
        }
        
        // Check required files
        if (!fileExists(config.intrinsic_file)) {
            std::cerr << "Error: Intrinsic file not found: " << config.intrinsic_file << std::endl;
            return -1;
        }
        if (!fileExists(config.extrinsic_file)) {
            std::cerr << "Error: Extrinsic file not found: " << config.extrinsic_file << std::endl;
            return -1;
        }
        
        // Ensure output directory exists
        size_t lastSlash = config.output_yaml.find_last_of('/');
        if (lastSlash != std::string::npos) {
            std::string dir = config.output_yaml.substr(0, lastSlash);
            if (!dir.empty() && !ensureDirectoryExists(dir)) {
                std::cerr << "Failed to create output directory" << std::endl;
                return -1;
            }
        }
        
        // Print configuration
        std::cout << "Configuration:\n"
                << "  Intrinsic file: " << config.intrinsic_file << "\n"
                << "  Extrinsic file: " << config.extrinsic_file << "\n"
                << "  Marker size: " << config.marker_size << "m\n"
                << "  Output YAML: " << config.output_yaml << "\n"
                << "  Visualization: " << config.visualization_path << "\n"
                << "  Marker IDs: " << config.expected_ids[0] << ", " 
                                    << config.expected_ids[1] << ", "
                                    << config.expected_ids[2] << ", "
                                    << config.expected_ids[3] << "\n"
                << "  Camera offset: " << config.camera_offset << "m\n"
                << "  Height above markers: " << config.height_above << "m\n"
                << "  Height below markers: " << config.height_below << "m\n";
        
        WorkspaceDefinition workspace(config.intrinsic_file, 
                                    config.extrinsic_file, 
                                    config.marker_size,
                                    config.expected_ids,
                                    config.camera_offset, 
                                    config.height_above, 
                                    config.height_below);

        if (workspace.defineWorkspace(config.output_yaml, config.visualization_path)) {
            std::cout << "Workspace definition successful!\n";
            return 0;
        }
        
        return -1;
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return -1;
    }
}