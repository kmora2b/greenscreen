import matplotlib.pyplot as plt
import cv2
import numpy as np

#Reads in greenscreen photo, converts it into RGB and returns as numpy array of (height,width,number of color channels)
#number of color channels is 3 because of RGB
def open_greensceen_img():
    img = cv2.imread('GreenScreen.png')
    
    cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows() # destroys the window showing image
    im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(type(img))
    print(img)

    return im_rgb

#Reads in background image, similar to open_greensceen_img()
def open_bg_img():
    background = cv2.imread('burning-house.jpg')
    
    cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows() # destroys the window showing image
    bg_rgb = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    return bg_rgb
    
#Displays images in matplotlib library which helps us look at how big the image is    
def display_image(f):
    plt.imshow(f, cmap='gray')
    plt.show()

#
def create_mask(img):
    #Convert from RGB to HSV (Hue Saturation Value) as it allows use to specifically get greens only while it is saturated or desaturated
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    #We get the range of green from dark green(lower_green) and cyan since we are using hsv values
    #With this range, we create a black and white mask that will only be of 2 colors of white and black
    #White is what we don't want in the image
    #Black is what we allow to show of the image
    lower_green = np.array([36,90,20])      
    cyan = np.array([70,255,255])
    mask = cv2.inRange(hsv, lower_green, cyan)
    
    #Created a masked image where whereever the mask is not black, the masked image will be turned to black as it is what we want to show of the image
    masked_image = np.copy(img)
    masked_image[mask != 0] = [0, 0, 0]

    return mask,masked_image

#Green can still occur in edges of mask so we should soften them to dissipate it and make the image look nicer    
def soften_edges(img):
    #Gets image and blurs it
    blurred_img = cv2.GaussianBlur(img, (21, 21), 0)
    
    #Initialize mask with an array of zeros
    mask = np.zeros(img.shape, np.uint8)
    
    #Convert image from RGB to Gray so to only handle 2 color channels of black and white
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #Gets the edges we found from the contour and draw black around them so to signal that it is an edge 
    cv2.drawContours(mask, contours, -1, (255,255,255),5)
    
    #Use the black and white mask, blurreed image, and original image as to finally get the edges and soften them
    #Wherever the mask is not black, we get the original image and only allow the blurred image to show through
    output = np.where(mask!=np.array([0, 0, 0]), blurred_img, img)
    return output

#Ours images will not be a perfect size so we should make it so that the mask, masked image, and background are the same size
#We will resize to the biggest image
def resized_imgs(background, mask, masked_image):
    #Create copies of the original images
    r_b, r_m, r_mi = np.copy(background), np.copy(mask), np.copy(masked_image)
    
    #Get the image sizes
    #shape is a 3d array that contains the height, width, and number of channels
    bg_size = background.shape
    masked_image_size = masked_image.shape
    
    if (bg_size > masked_image_size):
        r_b = cv2.resize(background, (bg_size[0], bg_size[1]))
        r_mi = cv2.resize(masked_image, (bg_size[0], bg_size[1]))
        r_m = cv2.resize(mask, (bg_size[0], bg_size[1]))
        
    elif (bg_size < masked_image_size):
        r_b = cv2.resize(background, (masked_image_size[0], masked_image_size[1]))
        r_m = cv2.resize(mask, (masked_image_size[0], masked_image_size[1]))

        r_mi = cv2.resize(masked_image, (masked_image_size[0], masked_image_size[1]))
    
    #Python unlike Java, can actually return multiple values at once
    return r_b, r_m, r_mi

#We get out mask and background and will finally compose out new image
def add_mask_2_background(background, mask,masked_image):
    #We resize our images and make new variables
    resized_bg, resized_mask, resized_mask_img = resized_imgs(background,mask,masked_image)
    print(resized_bg.shape)
    print(resized_mask.shape)
    print(resized_mask_img.shape)

    #We will use our mask to take out sections of it to allow our greenscreen image to be shown
    #Whenver the mask is black, the background image will turn those areas to black as well
    resized_bg[resized_mask == 0] = [0, 0, 0]
    
    #We combine our images by combining them and size the background image has areas where it is taken out and turned to black
    #We can easily place our masked image that was taken out of the greenscreen by turning the green pixels to black
    final_image = resized_bg + resized_mask_img
    
    #Lets resize it so that it looks nice and not a squished image
    resize_final_image = cv2.resize(final_image, (1024, 720))

    return resize_final_image

#Get greenscreen image
greenscreen = open_greensceen_img()
display_image(greenscreen)

#Get background image
background = open_bg_img()
display_image(background)

#Create mask and our greensceen image without the green pixels
mask = create_mask(greenscreen)[0]
masked_image = create_mask(greenscreen)[1]
display_image(mask)
display_image(masked_image)

#Creare our final image
final_image = add_mask_2_background(background, mask,masked_image)
display_image(final_image)