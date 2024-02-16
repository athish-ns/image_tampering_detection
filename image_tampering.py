# import cv2
# import numpy as np
# import mahotas.features
# from skimage import filters
# from skimage.transform import integral_image
# from sklearn.preprocessing import normalize
# from scipy.spatial.distance import cosine

# # Import NudeDetector from nudenet package
# from nudenet import NudeDetector

# def resize_image(image, target_size=(1000, 1000)):
#     """Resize an image to a target size."""
#     return cv2.resize(image, target_size)

# def compute_features(image):
#     """Compute various image features for tampering detection."""
#     # Resize the image to a smaller size
#     resized_image = resize_image(image)

#     # Compute HOG features
#     hog = cv2.HOGDescriptor()
#     hog_features = hog.compute(resized_image)

#     # Compute LBP features
#     lbp_features = mahotas.features.lbp(resized_image, radius=8, points=8)

#     # Compute gradient magnitude
#     gradient_magnitude = filters.sobel(resized_image)

#     # Compute integral image
#     integral_img = integral_image(resized_image)

#     # Normalize the features
#     hog_features = normalize(hog_features.reshape(1, -1))
#     lbp_features = normalize(lbp_features.reshape(1, -1))
#     gradient_magnitude = normalize(gradient_magnitude.reshape(1, -1))
#     integral_img = normalize(integral_img.reshape(1, -1))

#     # Combine all features into a single feature vector
#     feature_vector = np.concatenate((hog_features, lbp_features, gradient_magnitude, integral_img), axis=1)

#     return feature_vector.ravel()

# def detect_tampering(image_path, comparison_img_path):
#     # Load the images using OpenCV
#     img = cv2.imread(image_path)
#     comparison_img = cv2.imread(comparison_img_path)
    
#     if img is None or comparison_img is None:
#         print("Error loading images.")
#         return

#     # Resize images to a common size
#     img = resize_image(img)
#     comparison_img = resize_image(comparison_img)

#     # Convert the images to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     comparison_gray = cv2.cvtColor(comparison_img, cv2.COLOR_BGR2GRAY)
    
#     # Compute features for the original image
#     feature_vector = compute_features(gray)
    
#     # Compute features for the comparison image
#     feature_vector_comparison = compute_features(comparison_gray)
    
#     # Check if features are empty
#     if np.any(np.isnan(feature_vector)) or np.any(np.isnan(feature_vector_comparison)):
#         print("Error computing features. Check the input images.")
#         return
    
#     # Compute cosine similarity between the feature vectors
#     similarity_score = 1 - cosine(feature_vector, feature_vector_comparison)
    
#     print(f"Similarity score: {similarity_score}")
    
#     # You can set a threshold for the similarity score to determine tampering
#     threshold = 0.999
#     if similarity_score < threshold:
#         print("Tampering detected.")
#         # Analyze the extent of tampering and its nature (e.g., nudity, fake elements)
#         try:
#             analyze_tampering(img, comparison_img)
#             detect_bad_name(img, comparison_img)
#         except Exception as e:
#             print(f"An error occurred in tampering analysis: {e}")
#     else:
#         print("Images seem authentic.")

# def analyze_tampering(original_img, comparison_img):
#     # Analyze extent of tampering by calculating percentage difference in pixel values
    
#     # Convert images to grayscale for easier comparison
#     original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
#     comparison_gray = cv2.cvtColor(comparison_img, cv2.COLOR_BGR2GRAY)
    
#     # Calculate absolute difference between pixel values
#     abs_diff = cv2.absdiff(original_gray, comparison_gray)
    
#     # Count the number of non-zero pixels (i.e., pixels that are different)
#     num_different_pixels = np.count_nonzero(abs_diff)
    
#     # Calculate total number of pixels in the images
#     total_pixels = original_gray.size
    
#     # Calculate percentage difference
#     percentage_difference = (num_different_pixels / total_pixels) * 100
    
#     # Print the result
#     print(f"Percentage difference in pixel values: {percentage_difference:.2f}%")
    
#     # Perform nudity detection using NudeDetector
#     detector = NudeDetector()
#     try:
#         # Perform nudity detection on the images
#         original_result = detector.detect(original_img)
#         comparison_result = detector.detect(comparison_img)
        
#         # Check if either image is classified as containing nudity
#         if original_result['safe'] < 0.5 or comparison_result['safe'] < 0.5:
#             print("Tampering involves nudity.")
#             # Further analysis or actions can be added here, such as reporting or additional processing
#         else:
#             print("Tampering does not involve nudity.")
#     except Exception as e:
#         print(f"An error occurred in nudity detection: {e}")
    
#     # Return the percentage difference value for further analysis or reporting
#     return percentage_difference


# def detect_bad_name(original_img, comparison_img):
#     """Detect if the tampered image is used to bring a bad name to a person."""
#     # Perform text detection using the EAST text detector pre-trained on scene text detection
    
#     # Load the pre-trained EAST text detector
#     net = cv2.dnn.readNet("frozen_east_text_detection.pb")

#     # Define the input image dimensions
#     (H, W) = original_img.shape[:2]

#     # Define the new width and height and then determine the ratio in change
#     # for both the width and height
#     (newW, newH) = (320, 320)
#     rW = W / float(newW)
#     rH = H / float(newH)

#     # Resize the original image to the new dimensions
#     blob = cv2.dnn.blobFromImage(original_img, 1.0, (newW, newH),
#         (123.68, 116.78, 103.94), swapRB=True, crop=False)

#     # Pass the blob through the network and obtain the text detection
#     net.setInput(blob)
#     (scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

#     # Decode the predictions, then apply non-maxima suppression to suppress weak, overlapping bounding boxes
#     rects, confidences = decode_predictions(scores, geometry)
#     boxes = non_max_suppression(np.array(rects), probs=confidences)

#     # Initialize a list to store detected texts
#     detected_texts = []

#     # Loop over the bounding boxes
#     for (startX, startY, endX, endY) in boxes:
#         # Scale the bounding box coordinates based on the respective ratios
#         startX = int(startX * rW)
#         startY = int(startY * rH)
#         endX = int(endX * rW)
#         endY = int(endY * rH)
        
#         # Extract the region of interest (ROI) and convert it to grayscale
#         roi = original_img[startY:endY, startX:endX]
#         gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
#         # Threshold the ROI to segment the text
#         _, thresh = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
#         # Find contours in the thresholded image
#         contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         # Loop over the contours and extract the text
#         for contour in contours:
#             # Get the bounding box of the contour
#             x, y, w, h = cv2.boundingRect(contour)
#             # Extract the text region and append it to the detected texts list
#             detected_texts.append(gray_roi[y:y+h, x:x+w])

#     # Check if certain keywords or phrases are present in the detected text
#     bad_keywords = ["bad", "shame", "disgrace", "disgraceful", "shameful", "unethical", "dishonest"]
#     found_bad_name = False
#     for text in detected_texts:
#         for keyword in bad_keywords:
#             if keyword in text:
#                 found_bad_name = True
#                 break

#     if found_bad_name:
#         print("The tampered image may be used to bring a bad name to a person.")
#     else:
#         print("The tampered image does not appear to be used for bringing a bad name to a person.")

# # Utility function to decode the predictions from the EAST text detector
# def decode_predictions(scores, geometry):
#     (numRows, numCols) = scores.shape[2:4]
#     rects = []
#     confidences = []

#     for y in range(0, numRows):
#         scoresData = scores[0, 0, y]
#         xData0 = geometry[0, 0, y]
#         xData1 = geometry[0, 1, y]
#         xData2 = geometry[0, 2, y]
#         xData3 = geometry[0, 3, y]
#         anglesData = geometry[0, 4, y]

#         for x in range(0, numCols):
#             if scoresData[x] < 0.5:
#                 continue

#             (offsetX, offsetY) = (x * 4.0, y * 4.0)

#             angle = anglesData[x]
#             cos = np.cos(angle)
#             sin = np.sin(angle)

#             h = xData0[x] + xData2[x]
#             w = xData1[x] + xData3[x]

#             endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
#             endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
#             startX = int(endX - w)
#             startY = int(endY - h)

#             rects.append((startX, startY, endX, endY))
#             confidences.append(scoresData[x])

#     return rects, confidences

# # Utility function for non-maxima suppression
# def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
#     if len(boxes) == 0:
#         return []

#     if boxes.dtype.kind == "i":
#         boxes = boxes.astype("float")

#     pick = []

#     x1 = boxes[:, 0]
#     y1 = boxes[:, 1]
#     x2 = boxes[:, 2]
#     y2 = boxes[:, 3]

#     area = (x2 - x1 + 1) * (y2 - y1 + 1)
#     idxs = np.argsort(y2)

#     while len(idxs) > 0:
#         last = len(idxs) - 1
#         i = idxs[last]
#         pick.append(i)

#         xx1 = np.maximum(x1[i], x1[idxs[:last]])
#         yy1 = np.maximum(y1[i], y1[idxs[:last]])
#         xx2 = np.minimum(x2[i], x2[idxs[:last]])
#         yy2 = np.minimum(y2[i], y2[idxs[:last]])

#         w = np.maximum(0, xx2 - xx1 + 1)
#         h = np.maximum(0, yy2 - yy1 + 1)

#         overlap = (w * h) / area[idxs[:last]]

#         idxs = np.delete(idxs, np.concatenate(([last],
#             np.where(overlap > overlapThresh)[0])))

#     if probs is not None:
#         return boxes[pick].astype("int"), probs[pick]

#     return boxes[pick].astype("int")

# # Example usage

# image_path = "ok2.png"
# comparison_img_path = "ok.png"

# # Ensure image paths are converted to strings
# image_path = str(image_path)
# comparison_img_path = str(comparison_img_path)

# # Call the detect_tampering function with error handling
# try:
#     detect_tampering(image_path, comparison_img_path)
# except Exception as e:
#     print(f"An error occurred: {e}")

import cv2
import numpy as np
import tempfile
import pytesseract
import mahotas.features
from skimage import filters
from skimage.transform import integral_image
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine
from nudenet import NudeDetector

def resize_image(image, target_size=(1000, 1000)):
    return cv2.resize(image, target_size)

def compute_features(image):
    resized_image = resize_image(image)

    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(resized_image)

    lbp_features = mahotas.features.lbp(resized_image, radius=8, points=8)
    gradient_magnitude = filters.sobel(resized_image)
    integral_img = integral_image(resized_image)

    hog_features = normalize(hog_features.reshape(1, -1))
    lbp_features = normalize(lbp_features.reshape(1, -1))
    gradient_magnitude = normalize(gradient_magnitude.reshape(1, -1))
    integral_img = normalize(integral_img.reshape(1, -1))

    feature_vector = np.concatenate((hog_features, lbp_features, gradient_magnitude, integral_img), axis=1)

    return feature_vector.ravel()

def detect_tampering(image_path, comparison_img_path):
    img = cv2.imread(image_path)
    comparison_img = cv2.imread(comparison_img_path)
    
    if img is None or comparison_img is None:
        print("Error loading images.")
        return
    img = resize_image(img)
    comparison_img = resize_image(comparison_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    comparison_gray = cv2.cvtColor(comparison_img, cv2.COLOR_BGR2GRAY)
    
    feature_vector = compute_features(gray)
    
    feature_vector_comparison = compute_features(comparison_gray)
    
    if np.any(np.isnan(feature_vector)) or np.any(np.isnan(feature_vector_comparison)):
        print("Error computing features. Check the input images.")
        return
    
    similarity_score = 1 - cosine(feature_vector, feature_vector_comparison)
    
    print(f"Similarity score: {similarity_score}")
    
    threshold = 0.9999999
    if similarity_score < threshold:
        print("Tampering detected.")
        analyze_tampering(img, comparison_img)
        detect_bad_name(img, comparison_img)
    else:
        print("Images seem authentic.")

def analyze_tampering(original_img, comparison_img):
    original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    comparison_gray = cv2.cvtColor(comparison_img, cv2.COLOR_BGR2GRAY)
    
    abs_diff = cv2.absdiff(original_gray, comparison_gray)
    
    num_different_pixels = np.count_nonzero(abs_diff)
    
    total_pixels = original_gray.size
    
    percentage_difference = (num_different_pixels / total_pixels) * 100
    print(f"Percentage difference in pixel values: {percentage_difference:.2f}%")
    
    detector = NudeDetector()
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_original, \
             tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_comparison:
            cv2.imwrite(temp_original.name, original_img)
            cv2.imwrite(temp_comparison.name, comparison_img)
            
            original_result = detector.detect(temp_original.name)
            comparison_result = detector.detect(temp_comparison.name)
        
        if original_result[0]['unsafe'] or comparison_result[0]['unsafe']:
            print("Tampering involves nudity.")
        else:
            print("Tampering does not involve nudity.")
    except Exception as e:
        print(f"Person Detected but only with Male or Female Face (NudeNet consider this as a Nude factor)")
    
    return percentage_difference

def detect_bad_name(original_img, comparison_img):
    
    net = cv2.dnn.readNet("frozen_east_text_detection.pb")

    (H, W) = original_img.shape[:2]

    (newW, newH) = (320, 320)
    rW = W / float(newW)
    rH = H / float(newH)

    blob = cv2.dnn.blobFromImage(original_img, 1.0, (newW, newH),
        (123.68, 116.78, 103.94), swapRB=True, crop=False)

    net.setInput(blob)
    (scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

    rects, confidences = decode_predictions(scores, geometry)
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    detected_texts = []

    for (startX, startY, endX, endY) in boxes:
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        
        roi = original_img[startY:endY, startX:endX]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        _, thresh = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            detected_texts.append(gray_roi[y:y+h, x:x+w])
    bad_keywords = ["bad", "shame", "disgrace", "disgraceful", "shameful", "unethical", "dishonest"]
    found_bad_name = False
    for text in detected_texts:
        for keyword in bad_keywords:
            if keyword in text:
                found_bad_name = True
                break

    if found_bad_name:
        print("The tampered image may be used to bring a bad name to a person.")
    else:
        print("The tampered image does not appear to be used for bringing a bad name to a person.")
def decode_predictions(scores, geometry):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            if scoresData[x] < 0.5:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return rects, confidences
def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    if probs is not None:
        return boxes[pick].astype("int"), probs[pick]

    return boxes[pick].astype("int")
image_path = input("Enter the real image path:")
comparison_img_path = input("Enter the suspicious image path:")
detect_tampering(image_path, comparison_img_path)
