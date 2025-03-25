import cv2
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description='Generate ArUco markers for workspace definition')
    parser.add_argument('--dict', type=int, default=0, help='ArUco dictionary ID (default: 0 for DICT_4X4_50)')
    parser.add_argument('--start', type=int, default=0, help='Starting marker ID (default: 0)')
    parser.add_argument('--count', type=int, default=4, help='Number of markers to generate (default: 4)')
    parser.add_argument('--size', type=int, default=600, help='Output marker size in pixels (default: 600)')
    parser.add_argument('--border', type=int, default=1, help='Border size in bits (default: 1)')
    parser.add_argument('--output', type=str, default='marker', help='Output filename prefix (default: "marker")')
    
    args = parser.parse_args()
    
    # Define dictionary names for different OpenCV versions
    dictionaries = [
        cv2.aruco.DICT_4X4_50,
        cv2.aruco.DICT_5X5_50,
        cv2.aruco.DICT_6X6_50,
        cv2.aruco.DICT_7X7_50
    ]
    
    dictionary_id = args.dict
    if dictionary_id < 0 or dictionary_id >= len(dictionaries):
        print(f"Error: Dictionary ID must be between 0 and {len(dictionaries)-1}")
        return
    
    # Create ArUco dictionary - method varies by OpenCV version
    try:
        # Newest versions of OpenCV
        dictionary = cv2.aruco.getPredefinedDictionary(dictionaries[dictionary_id])
    except AttributeError:
        # Try alternative approach
        try:
            dictionary = cv2.aruco.Dictionary.create(dictionaries[dictionary_id])
        except (AttributeError, TypeError):
            print("Error: Could not create ArUco dictionary with your OpenCV version")
            print("Your OpenCV version:", cv2.__version__)
            return
    
    # Generate markers
    for i in range(args.start, args.start + args.count):
        # Create empty image
        marker_img = np.zeros((args.size, args.size), dtype=np.uint8)
        
        # Try different marker generation methods based on OpenCV version
        try:
            # Newest method
            cv2.aruco.generateImageMarker(dictionary, i, args.size, marker_img, args.border)
        except AttributeError:
            try:
                # Older method - create and draw
                marker_img = cv2.aruco.drawMarker(dictionary, i, args.size, borderBits=args.border)
            except AttributeError:
                # Even older versions
                marker_params = cv2.aruco.DetectorParameters_create()
                marker_img = cv2.aruco.drawMarker(dictionary, i, args.size, marker_params, args.border)
                
        # Add a white border around the marker
        border_size = args.size // 10
        bordered_img = cv2.copyMakeBorder(marker_img, 
                                         border_size, border_size, border_size, border_size, 
                                         cv2.BORDER_CONSTANT, value=255)
        
        # Add marker ID text
        full_img = cv2.copyMakeBorder(bordered_img, 
                                    border_size, 0, 0, 0, 
                                    cv2.BORDER_CONSTANT, value=255)
        
        text = f"Marker ID: {i}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 1, 1)[0]
        text_x = (full_img.shape[1] - text_size[0]) // 2
        cv2.putText(full_img, text, (text_x, border_size // 2 + text_size[1]), 
                   font, 1, 0, 2, cv2.LINE_AA)
        
        filename = f"{args.output}_{i}.png"
        cv2.imwrite(filename, full_img)
        print(f"Generated marker ID {i}: {filename}")

if __name__ == "__main__":
    main()