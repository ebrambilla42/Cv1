#Group Members: Emilio Brambilla, Moritz Lahann, Lasse Haffke
from shutil import copy
import cv2 as cv
import numpy as np

PATH_READ = '/Users/emibrambilla/Desktop/courses/3_cv1/Assignemnts_cv1/1/img/visual_attention.png'
PATH_WRITE = '/Users/emibrambilla/Desktop/'

#load image 
image = cv.imread(PATH_READ)

#1. converting to gray scale 
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

#2. Compute the integral image,
img_integral = cv.integral(gray)

#3. For every pixel, get the center-surround windows and compute the average based on the integral image for 
#each window (hint: check slides 23 & 24 in the lecture). 
# Use sizes 31 × 31 for the center and 41 × 41 surround windows.

def win_avg(win_size, img_integral, curr_x, curr_y):
    #Avg = D-C-B+A
    A = img_integral[curr_x -1][curr_y -1]
    B = img_integral[curr_x + win_size -1][curr_y -1]
    C = img_integral[curr_x -1][curr_y + win_size -1]
    D = img_integral[curr_x + win_size -1][curr_y + win_size -1]

    #multiply win_size per 7 for a better img result
    avg = (D-C-B+A)/(win_size * 7) 
    return avg

#sliding center-surround windows
def sliding_center_surround_windows(img_integral, center_win_size, surround_win_size):
    #iniciate saliency map
    saliency_img = img_integral.copy()

    for (x, y),value in np.ndenumerate(img_integral):
        try:
            #calculate the average (lesson formula)
            avg_center_window = win_avg(center_win_size, img_integral, x, y)
            avg_surround_window = win_avg(surround_win_size, img_integral, x, y)

            #4. Subtract the center-surround averages to compute a saliency value. 
            saliency_value = avg_surround_window - avg_center_window

            #5. Construct the saliency map by placing the saliency value of that pixel in its corresponding 
            # position in a new matrix.
            saliency_img[x][y] = saliency_value 
        except:
            #padding
            saliency_img[x][y] = saliency_img[x-1][y-1]

    return saliency_img


#window sizes
center_window_size, surround_window_size = 31, 41
small_center_window_size, small_surround_window_size = 10, 20
big_center_window_size, big_surround_window_size = 100, 150

#construct saliency
saliency_img = sliding_center_surround_windows(img_integral, center_window_size, surround_window_size)
small_saliency_img = sliding_center_surround_windows(img_integral, small_center_window_size, small_surround_window_size)
big_saliency_img = sliding_center_surround_windows(img_integral, big_center_window_size, big_surround_window_size)

#6. Display the saliency map 
cv.imwrite(PATH_WRITE + 'saliency.png',saliency_img)

#Construct and show the new saliency map smaller and larger center-surround windows
cv.imwrite(PATH_WRITE + 'small_saliency.png',small_saliency_img)
cv.imwrite(PATH_WRITE + 'big_saliency.png', big_saliency_img)




