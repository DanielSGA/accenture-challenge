# USAGE
# python mask_rcnn.py --mask-rcnn mask-rcnn-coco --image images/example_01.jpg
# python mask_rcnn.py --mask-rcnn mask-rcnn-coco --image images/P6040048.jpg  --visualize 1

# import the necessary packages
import numpy as np
import argparse
import random
import time
import cv2
import os
from sklearn.cluster import KMeans
from collections import Counter
import json

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-m", "--mask-rcnn", required=True,
	help="base path to mask-rcnn directory")
ap.add_argument("-v", "--visualize", type=int, default=0,
	help="whether or not we are going to visualize each instance")
ap.add_argument("-c", "--confidence", type=float, default=0.9,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.6,
	help="minimum threshold for pixel-wise mask segmentation")
args = vars(ap.parse_args())

# load the COCO class labels our Mask R-CNN was trained on
labelsPath = os.path.sep.join([args["mask_rcnn"],
	"object_detection_classes_coco.txt"])
LABELS = open(labelsPath).read().strip().split("\n")

# load the set of colors that will be used when visualizing a given
# instance segmentation
colorsPath = os.path.sep.join([args["mask_rcnn"], "colors.txt"])
COLORS = open(colorsPath).read().strip().split("\n")
COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
COLORS = np.array(COLORS, dtype="uint8")

# derive the paths to the Mask R-CNN weights and model configuration
weightsPath = os.path.sep.join([args["mask_rcnn"],
	"frozen_inference_graph.pb"])
configPath = os.path.sep.join([args["mask_rcnn"],
	"mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])

with open('file.json') as f:
        data = json.load(f)
# load our Mask R-CNN trained on the COCO dataset (90 classes)
# from disk
print("[INFO] loading Mask R-CNN from disk...")
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

# load our input image and grab its spatial dimensions
image = cv2.imread(args["image"])
(H, W) = image.shape[:2]

# construct a blob from the input image and then perform a forward
# pass of the Mask R-CNN, giving us (1) the bounding box  coordinates
# of the objects in the image along with (2) the pixel-wise segmentation
# for each specific object
blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
(boxes, masks) = net.forward(["detection_out_final", "detection_masks"])
end = time.time()

# show timing information and volume information on Mask R-CNN
print("[INFO] Mask R-CNN took {:.6f} seconds".format(end - start))
print("[INFO] boxes shape: {}".format(boxes.shape))
print("[INFO] masks shape: {}".format(masks.shape))

# loop over the number of detected objects
for i in range(0, boxes.shape[2]):
	# extract the class ID of the detection along with the confidence
	# (i.e., probability) associated with the prediction
	classID = int(boxes[0, 0, i, 1])
	confidence = boxes[0, 0, i, 2]

	# filter out weak predictions by ensuring the detected probability
	# is greater than the minimum probability
	if confidence > args["confidence"]:
		# clone our original image so we can draw on it
		clone = image.copy()

		# scale the bounding box coordinates back relative to the
		# size of the image and then compute the width and the height
		# of the bounding box
		box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
		(startX, startY, endX, endY) = box.astype("int")
		boxW = endX - startX
		boxH = endY - startY

		# extract the pixel-wise segmentation for the object, resize
		# the mask such that it's the same dimensions of the bounding
		# box, and then finally threshold to create a *binary* mask
		mask = masks[i, classID]
		mask = cv2.resize(mask, (boxW, boxH),
			interpolation=cv2.INTER_NEAREST)
		mask = (mask > args["threshold"])

		# extract the ROI of the image
		roi = clone[startY:endY, startX:endX]

		# check to see if are going to visualize how to extract the
		# masked region itself
		if args["visualize"] > 0:
			# convert the mask from a boolean to an integer mask with
			# to values: 0 or 255, then apply the mask
			visMask = (mask * 255).astype("uint8")
			instance = cv2.bitwise_and(roi, roi, mask=visMask)

			# show the extracted ROI, the mask, along with the
			# segmented instance
			cv2.imshow("ROI", roi)
			cv2.imshow("Mask", visMask)
			cv2.imshow("Segmented", instance)






import os
import DetectChars
import DetectPlates
import PossiblePlate

# module level variables ##########################################################################
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = False

###################################################################################################
def main():

    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()         # attempt KNN training

    if blnKNNTrainingSuccessful == False:                               # if KNN training was not successful
        print("\nerror: KNN traning was not successful\n")  # show error message
        return                                                          # and exit program
    # end if

    imgOriginalScene  = roi                # open image

    if imgOriginalScene is None:                            # if image was not read successfully
        print("\nerror: image not read from file \n\n")  # print error message to std out
        os.system("pause")                                  # pause so user can see error message
        return                                              # and exit program
    # end if

    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)           # detect plates

    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)        # detect chars in plates

    cv2.imshow("imgOriginalScene", imgOriginalScene)            # show scene image

    if len(listOfPossiblePlates) == 0:                          # if no plates were found
        print("\nno license plates were detected\n")  # inform user no plates were found
    else:                                                       # else
                # if we get in here list of possible plates has at leat one plate

                # sort the list of possible plates in DESCENDING order (most number of chars to least number of chars)
        listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

                # suppose the plate with the most recognized chars (the first plate in sorted by string length descending order) is the actual plate
        licPlate = listOfPossiblePlates[0]

        if len(licPlate.strChars) == 0:                     # if no chars were found in the plate
            print("\nno characters were detected\n\n")  # show message
            return                                          # and exit program
        # end if

        drawRedRectangleAroundPlate(imgOriginalScene, licPlate)             # draw red rectangle around plate

        print("\nlicense plate read from image = " + licPlate.strChars + "\n")  # write license plate text to std out
        print("----------------------------------------")
        
        if licPlate.strChars in data['people'][0].values():
                flagPlate = 0
                
        elif licPlate.strChars in data['people'][1].values():
                flagPlate = 1
                
        elif licPlate.strChars in data['people'][2].values():
                flagPlate = 2
                
        elif licPlate.strChars in data['people'][3].values():
                flagPlate =3
        
        else:
                print("El guardia vendra enseguida a verificar sus datos")
                exit(1)
 
        

        writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)           # write license plate text on the image

        cv2.imshow("imgOriginalScene", imgOriginalScene)                # re-show scene image

        filename = licPlate.strChars
        filename= str(filename) + ".jpg"

        cv2.imwrite(filename, imgOriginalScene)           # write image out to file

    # end if else
    from sklearn.cluster import KMeans
    from collections import Counter


    hexa2= get_dominant_color(roi, k=8, image_processing_size = None)

    if (flagPlate == hexa2): 
            print("Bienvenido seas " +  data['people'][hexa2]['Name']+ " , que tengas un excelente día." +
                  "Tu lugar de estacionamineto es el: " + str(random.randint(1,545)))
    else:
            print("El guardia vendra enseguida a verificar sus datos")

    cv2.waitKey(0)					# hold windows open until user presses a key

    return
# end main

###################################################################################################
def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):

    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)            # get 4 vertices of rotated rect

    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)         # draw 4 red lines
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)
# end function

###################################################################################################
def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0                             # this will be the center of the area the text will be written to
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0                          # this will be the bottom left of the area that the text will be written to
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX                      # choose a plain jane font
    fltFontScale = float(plateHeight) / 30.0                    # base font scale on height of plate area
    intFontThickness = int(round(fltFontScale * 1.5))           # base font thickness on font scale

    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale, intFontThickness)        # call getTextSize

            # unpack roatated rect into center point, width and height, and angle
    ( (intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg ) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)              # make sure center is an integer
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX)         # the horizontal location of the text area is the same as the plate

    if intPlateCenterY < (sceneHeight * 0.75):                                                  # if the license plate is in the upper 3/4 of the image
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))      # write the chars in below the plate
    else:                                                                                       # else if the license plate is in the lower 1/4 of the image
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))      # write the chars in above the plate
    # end if

    textSizeWidth, textSizeHeight = textSize                # unpack text size width and height

    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))           # calculate the lower left origin of the text area
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))          # based on the text area center, width, and height

            # write the text on the image
    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace, fltFontScale, SCALAR_YELLOW, intFontThickness)
# end function
##################################################################################
def get_dominant_color(imgOriginalScene, k=2, image_processing_size = (25,25)):
        """
        takes an image as input
        returns the dominant color of the image as a list
        
        dominant color is found by running k means on the 
        pixels & returning the centroid of the largest cluster

        processing time is sped up by working with a smaller image; 
        this resizing can be done with the image_processing_size param 
        which takes a tuple of image dims as input

        >>> get_dominant_color(my_image, k=4, image_processing_size = (25, 25))
        [56.2423442, 34.0834233, 70.1234123]
        """
        #resize image if new dims provided
        if image_processing_size is not None:
            image = cv2.resize(imgOriginalScene, image_processing_size, 
                                interpolation = cv2.INTER_AREA)
        
        #reshape the image to be a list of pixels
        image = imgOriginalScene.reshape((imgOriginalScene.shape[0] * imgOriginalScene.shape[1], 3))

        #cluster and assign labels to the pixels 
        clt = KMeans(n_clusters = k)
        labels = clt.fit_predict(image)

        #count labels to find most popular
        label_counts = Counter(labels)

        #subset out most popular centroid
        dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]

        mylist = dominant_color
        myorder = [2, 1, 0]
        mylist = [mylist[i] for i in myorder]
            

        hexa ="#{:02x}{:02x}{:02x}".format(int(mylist[0]), int(mylist[1]), int(mylist[2]))
        
        
        if str(hexa) in data['people'][0].values():
                flagHEX = 0
                
        elif str(hexa) in data['people'][1].values():
                flagHEX = 1
                
        elif str(hexa) in data['people'][2].values():
                flagHEX = 2
               
        elif str(hexa) in data['people'][3].values():
                flagHEX = 3
                
        else:
                print("El guardia vendra enseguida a verificar sus datos")
            
        return(flagHEX)
                


        
        


###################################################################################################
if __name__ == "__main__":
    main()

##		# now, extract *only* the masked region of the ROI by passing
##		# in the boolean mask array as our slice condition
##		roi = roi[mask]
##
##		# randomly select a color that will be used to visualize this
##		# particular instance segmentation then create a transparent
##		# overlay by blending the randomly selected color with the ROI
##		color = random.choice(COLORS)
##		blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")
##
##		# store the blended ROI in the original image
##		clone[startY:endY, startX:endX][mask] = blended
##
##		# draw the bounding box of the instance on the image
##		color = [int(c) for c in color]
##		cv2.rectangle(clone, (startX, startY), (endX, endY), color, 2)
##
##		# draw the predicted label and associated probability of the
##		# instance segmentation on the image
##		text = "{}: {:.4f}".format(LABELS[classID], confidence)
##		cv2.putText(clone, text, (startX, startY - 5),
##			cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
##
##		# show the output image
##		cv2.imshow("Output", clone)
##		cv2.waitKey(0)
