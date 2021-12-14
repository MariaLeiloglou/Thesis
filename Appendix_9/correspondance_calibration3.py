import cv2
import os
import numpy as np

# Open video file
rootdir = os.path.dirname(os.path.realpath('__file__'))
# name = rootdir + '\\' + 'Data' + '\\' + 'hand' + '\\' + 'valleyHand_0'
name = r'/Users/marialeiloglou/Desktop/appearance2/appearance_2.avi'
cap = cv2.VideoCapture(name)
i = 1
j = 1 # fibre label
tracked_spots = 0
radi = [] # we find in the difference image in every loop
centers= []


DELTA_BACKG_INTENS = 5
x = 0
y = 0

while (cap.isOpened()):
    ret, frame = cap.read()# read frame
    if frame is None:
        break
    if j == 1:
        rows = frame.shape[0] # read size of frame
        columns = frame.shape[1] # read size of frame
        mask = np.zeros((rows, columns)) # prepare mask with fibres

    if ret == True:
        cv2.imwrite('img' + '%05d' % i + '.png', frame) # save frame
        cv2.imshow('frame', frame)
        i += 1
        if i >= 21: # in this video spots appear after frame 20, i = 21 /current frame is 20

            # get the difference in frames
            image1 = frame

            k = i - 2
            image2 = cv2.imread('img' + '%05d' % k + '.png')
            difference1 = cv2.subtract(image1, image2) # subtract frames
            difference1[mask != 0] = 0 # remove fibres already found


            # noise removal
            difference = cv2.cvtColor(difference1, cv2.COLOR_BGR2GRAY)
            immedian = cv2.medianBlur(difference, 5)
            imgdenoise = cv2.fastNlMeansDenoising(immedian, None, 10, 7, 21)

            hist = cv2.calcHist([imgdenoise], [0], None, [256],
                                [0, 256])
            intensity_bg = np.argmax(hist)
            ret, im_gray_no_backg = cv2.threshold(imgdenoise, intensity_bg + DELTA_BACKG_INTENS, 255,
                                                  cv2.THRESH_TOZERO)

            circles = cv2.HoughCircles(im_gray_no_backg, cv2.HOUGH_GRADIENT, 1, 30, param1=5, param2=20, minRadius=11,
                                       maxRadius=25) # find circles in the image
            if circles is not None:
                for c in circles[0, :]: # for each circle found
                  center = (int(c[0]), int(c[1])) # find centre
                  if i == 21: # for the first fibre in the video
                      if mask[(int(c[1]), int(c[0]))] == 0: # make sure does not coincide with previous spot
                          centers.append((c[1], c[0])) # save the spot centre
                          radius = c[2]
                          radi.append(radius) # save the spot radius
                          mask = cv2.circle(mask, center, int(radius), j, -1) # save the spot in the mask
                          cv2.imshow("mask", mask)
                          cv2.imwrite('fibre' + '%05d' % j + '.png', mask) # save current mask
                          j += 1
                  elif i != 21: # for the rest of the fibres in the video
                      if j < 173:# make sure you haven't found all fibres in the video
                          if mask[(int(c[1]), int(c[0]))] == 0:# make sure does not coincide with previous spot
                              centers.append((c[1], c[0]))# save the spot centre
                              radius = c[2]
                              radi.append(radius)# save the spot radius
                              mask = cv2.circle(mask, center, int(radius), j, -1)# save the spot in the mask
                              cv2.imshow("mask", mask)
                              cv2.imwrite('fibre' + '%05d' % j + '.png', mask)# save current mask
                              j += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()


id = 0

# show labeling with the mask
for c in centers:
   id += 1
   labels = cv2.putText(mask, str(id), (int(round(c[1]) - 10), int(round(c[0]) + 10)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

cv2.imwrite(r'all spots.png', labels)# save labeled mask

# save calibration file

np.savetxt('centres.txt', centers, fmt='%d')
np.savetxt('centreradi.txt', radi, fmt='%d')




cv2.destroyAllWindows()


