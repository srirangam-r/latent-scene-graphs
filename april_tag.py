import cv2
import cv2.aruco as aruco

# 1. Define the dictionary (Tag36h11)
april_tag_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)

# 2. Generate the marker
# parameters: dictionary, tag ID (e.g., 1), size in pixels (e.g., 200)
img = aruco.generateImageMarker(april_tag_dict, 1, 500)

# 3. Save it
cv2.imwrite("tag36h11_id1.png", img)