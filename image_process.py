# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
# import os

# #get directory path
# dir_path = os.path.dirname(os.path.realpath(__file__))
# dir_path = dir_path.replace('\\','/')

# #the paths to the data
# path_lst = ['Bees','MurderHornets']
# #create path
# path_bees = dir_path + '/Data/' + path_lst[0]
# path_hornets = dir_path + '/Data/' + path_lst[1]


# def remove_backgroundManual():
#     '''
#         remove background of an image
#     ''' 

#     for bee_img in os.listdir(path_bees):
#         #for each bee_img name, join it to the path of bees to read it
#         image_array = cv2.imread(os.path.join(path_bees,bee_img), cv2.IMREAD_COLOR)

#         #get the gray scale image
#         gray_img = cv2.cvtColor(image_array,cv2.COLOR_BGR2GRAY)

#         #find threshold
#         _, thresh = cv2.threshold(gray_img,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

#         #find image contours
#         img_contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2]

#         #sort the contours
#         img_contours = sorted(img_contours,key=cv2.contourArea)

#         for i in img_contours:
#             if cv2.contourArea(i) > 100:
#                 break
        
#         #generate the mask using np.zeros
#         mask = np.zeros(image_array.shape[:2],np.uint8)

#         #draw contours
#         cv2.drawContours(mask, [i],-1,255,-1)

#         #apply bitwise operator
#         new_img = cv2.bitwise_and(image_array,image_array,mask=mask)

#         plt.imshow(new_img)
#         plt.show()
        
#         cv2.waitKey(2)

# def remove_backgroundManual():
#     '''
#         remove background of an image
#     ''' 

#     for bee_img in os.listdir(path_bees):
#         #for each bee_img name, join it to the path of bees to read it
#         image_array = cv2.imread(os.path.join(path_bees,bee_img))

#         #change color of foreground to RGB and resize image to match shape of R-band in RGB outputmap
#         image_array = cv2.cvtColor(image_array,cv2.COLOR_BGR2RGB)
#         image_array = cv2.resize(image_array,r.shape[1],r.shape[0])
        

#         plt.imshow(new_img)
#         plt.show()
        
#         cv2.waitKey(2)

# remove_backgroundManual()