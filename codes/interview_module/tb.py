from functions.helper import extract_face 
import cv2
import dlib
dnn_net = cv2.dnn.readNetFromCaffe("models/deploy.prototxt", "models/res10_300x300_ssd_iter_140000.caffemodel")

predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
img=cv2.imread('raph.jpg')
# Example usage
mouth_image = extract_face(img,dnn_net,predictor)
# print(mouth_image.shape)
if mouth_image is not None:
    cv2.imshow("Mouth Region", mouth_image[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
