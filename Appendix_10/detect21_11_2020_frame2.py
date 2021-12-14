# Importing necessary packages for this project
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import glob
from scipy.io import loadmat
from scipy.spatial import distance
from scipy import interpolate
from matplotlib import pyplot as plt
import time

begin = time.time()

# Function for image brighness control
def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


# read image -- get the 1 channel for conversion to grayscale
img1 = cv2.imread('mono2.png')

# Change image brighness-optional-
# img = increase_brightness(img, value=20)

# Change from RGB to gray
img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# cv2.imwrite('./SpotPatternDATA/trial3_mountain/image_7_proc.png', img)

# Apply Gaussian and Median blurring
img = cv2.GaussianBlur(img, (5, 5), 0)
cimg = cv2.medianBlur(img, 5)

# Get number of image rows
rows = img.shape[0]

# Morphological operations -- erosion/detach two connected objects  & dilation/increases our shrinked spot
kernel = np.ones((3, 3), np.uint8)
# img = cv2.erode(cimg, kernel, iterations=1)
img = cv2.dilate(cimg, kernel, iterations=2)

# Apply hough circles transform to get the circles on image
circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 30, param1=5, param2=20, minRadius=10, maxRadius=23) # rows/24

# Get the number of detected circles
circles = np.uint16(np.around(circles))

# file = open("circles.txt", 'w')
centers = []
radi = []
# Iterate over the detected circles
for i in circles[0,:]:
	# Get circle centre position
    center = (i[0], i[1])
    centers.append(center)
    # Draw centre of the detected circle on input image
    cv2.circle(img, center, 1, (0, 100, 100), 3)
    # Get circle radius
    radius = i[2]
    radi.append(radius)
    # Draw perimeter of detected cirle on input image
    cv2.circle(img, center, radius, (255, 0, 255), 3)
    # cv2.circle(img, (i[0], i[1]), i[2], (0,255,0), 2)
    # cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
    # file.write(str(i[0]) + ", " + str(i[1]) + ", " + str(i[2]) + "\n")

centers = np.array(centers)
radi = np.array(radi)
nondetected = 172 - len(radi) # the spots that have not been detected



#find the weighted centre of the segmented spots
wc1sum = [sum(x) for x in zip(*centers)]# sum of x/y locations of centres of spots


wc1 = np.zeros((2, 1), dtype=float)
for i, element in enumerate(wc1sum):
 wc1[i] = element / len(radi) # average of x/y locations of centres of spots in frame 2







# example with non-detected labels
nondetectedlabels = [6, 7, 19, 36, 107, 112, 136, 138, 139, 153, 154, 155, 157, 161]

#import reference map for frame 1
labels = np.loadtxt('labels.txt', dtype=float)
calibMaskold = np.zeros((172, 2), dtype=float)
calibMaskold[:, 0] = np.loadtxt('centrex.txt', dtype=float)
calibMaskold[:, 1] = np.loadtxt('centrey.txt', dtype=float)
spotDiam = np.loadtxt('centreradi.txt', dtype=float)
meanDiam = np.mean(spotDiam)

#find the weighted centre of the labels
wc2sum = [sum(x) for x in zip(*calibMaskold)]# sum of x/y locations of centres of spots inn frame 1

wc2 = np.zeros((2, 1), dtype=float)
for i, element in enumerate(wc2sum):
 wc2[i] = element / len(calibMaskold)# average of x/y locations of centres of spots in frame 1



#find the weighted centre differences
dwc21= np.zeros((2, 1), dtype=float)
dwc21[0] = wc1[0]-wc2[0]
dwc21[1] = wc1[1]-wc2[1]



#correct the label locations in the frame 1
calibMask = np.zeros((172, 2), dtype=float)
calibMask[:, 0] = calibMaskold[:, 0] + dwc21[0]
calibMask[:, 1] = calibMaskold[:, 1] + dwc21[1]

#match the spots we found in frame 2 with the labeled in frame 1 (corrected for translation)
NumList = []
idofspot = []
matchedspots = []
matcheddistances = []

distanceandid = np.zeros((172, 2), dtype=int)
distances2keep = 3 #keep the closest three neighbour spots


list_1d_labels = [] #the detected identification labels
list_2d_x = []
list_2d_y = []
list_2d_r = []
list_2d_d = []
for i in range(len(calibMask)):#iterate through the corrected reference mask spots of frame 1
    list_1d_labels.append(i + 1)
    distances_to_i = []
    for j in range(len(centers)):#iterate through the segmented spots of frame 2
        dists = distance.euclidean(centers[j], calibMask[i]) #calculate distance
        distances_to_i.append(dists) #save distances
    sorted_ind = np.argsort(distances_to_i) #sort distances from smaller to larger
    list_x = []
    list_y = []
    list_r = []
    list_d = []
    for k in range(distances2keep): # 1,2,3
        ind_tmp = sorted_ind[k] # find corresponding segmented spot id
        x_tmp, y_tmp = centers[ind_tmp] # find corresponding segmented spot centre x/y locations
        list_x.append(x_tmp) # save corresponding segmented spot centre x
        list_y.append(y_tmp) # save corresponding segmented spot centre y
        list_r.append(radi[ind_tmp]) # save corresponding segmented spot radius
        list_d.append(distances_to_i[ind_tmp]) # save corresponding distance to reference spot in frame 1
    list_2d_x.append(list_x)  # save for all three spots centre x
    list_2d_y.append(list_y)  # save for all three spots centre y
    list_2d_r.append(list_r)  # save for all three spots radius
    list_2d_d.append(list_d)  # save for all three spots distance to reference spot in frame 1


#from each label neighborhood keep the closest neighboor out of three
closestd = []
closestx = []
closesty = []
closestr = []
closestlabels = []

for i in range(0, len(list_1d_labels)):#iterate through the corrected reference mask spots of frame 1
  label_tmp = i+1
  if not label_tmp in nondetectedlabels: #example with missing spots
                distancematch = list_2d_d[i][0]
                centrexmatch = list_2d_x[i][0]
                centreymatch = list_2d_y[i][0]
                radiusmatch = list_2d_r[i][0]
                labelmatch = i+1
                closestd.append(distancematch)
                closestx.append(centrexmatch)
                closesty.append(centreymatch)
                closestr.append(radiusmatch)
                closestlabels.append(labelmatch)

# add the missing spots
#centresx = [840.532775793666, 389.019388979401, 542.192792428873, 743, 569, 642.153911421051]
#centresy = [453.008692742219, 382.503457385869, 134.164744158939, 221, 170, 129.492924752580]
#radi = [13.8022027870862, 12.1012218582752, 11.1676171307841, 15, 15, 20.0951315537014]

#for i, label in enumerate(nondetectedlabels):
                #    closestlabels.append(label)
                #   closestx.append(centresx[i])
                #   closesty.append(centresy[i])
#    closestr.append(radi[i])

# show the nearest from the neighborhood of three
for j in range(0, len(closestlabels)):
    img2 = cv2.putText(img1, str(closestlabels[j]), (int(round(closestx[j]) - 10), int(round(closesty[j]) + 10)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

cv2.imshow('label closest neighbors', img2)
cv2.waitKey(4000)
cv2.destroyAllWindows()
cv2.imwrite(r'closest neighbours.png', img2) # save the image with the tracked spots and their identification labels



#   show the previous and the shifted labels and the segmented spots in the new image

#   show the reference labels
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for i in range(0, len(calibMask)): #iterate through the reference mask centres of spots
   img4 = cv2.line(img, (int(round(calibMaskold[i, 0])), int(round(calibMaskold[i, 1]))), (int(round(calibMask[i, 0])), int(round(calibMask[i, 1]))), (0, 255, 0), 1)


cv2.imshow('shifted labels', img4) # shows movement (line) of reference spots centres from frame 1 after the translation correction applied
cv2.waitKey(4000)
cv2.destroyAllWindows()
cv2.imwrite(r'shifted labels.png', img4)
#   there are 14 labels missing [6, 7, 19, 36, 107, 112, 136, 138, 139, 153, 154, 155, 157, 161]:


end = time.time()
print(f"Total runtime (seconds) of the program is {end - begin}")
# Total runtime (seconds) of the program is 1.090383768081665
exit()








