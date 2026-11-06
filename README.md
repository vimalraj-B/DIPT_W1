# Workshop 1 – Adding Sunglasses to Your Passport Photo Using OpenCV
### Developed By : VIMALRAJ B
### Register Number : 212224230304
### Aim :
To add sunglasses to a passport-size face image using Python and OpenCV by applying image resizing, alpha masking, and arithmetic operations.
### Requirements :
Hardware Requirements:
- Computer or Laptop
- Minimum 4 GB RAM
- Webcam or input face image
Software Requirements:
- Python
- Jupyter Notebook
- OpenCV
- NumPy
- Matplotlib
### Procedure :
1. Import the required libraries: OpenCV, NumPy, and Matplotlib.
2. Load the face image using cv2.imread().
3. Load the sunglasses image with its alpha channel.
4. Resize the sunglasses image to fit the eye region.
5. Separate the colour channels and alpha channel.
6. Create a copy of the original face image.
7. Select the eye region (ROI) in the face image.
8. Apply the mask using arithmetic operations to blend the sunglasses with the face.
9. Replace the original eye region with the final blended image.
10. Display the original image and the final image with sunglasses using Matplotlib.


### Program :
```python
# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Load the Face Image
faceImage = cv2.imread('ex.jpg')
plt.imshow(faceImage[:,:,::-1]);plt.title("Face")
```
```python
faceImage.shape
#resized_faceImage.shape
faceImage.shape
# Load the Sunglass image with Alpha channel
# (http://pluspng.com/sunglass-png-1104.html)
glassPNG = cv2.imread('sung3.png',-1)
plt.imshow(glassPNG[:,:,::-1]);
plt.title("glassPNG")
```
```python
# Resize the image to fit over the eye region
glassPNG = cv2.resize(glassPNG,(200,80))
print("image Dimension ={}".format(glassPNG.shape))
# Separate the Color and alpha channels
glassBGR = glassPNG[:,:,0:3]
glassMask1 = glassPNG[:,:,3]# Display the images for clarity
plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(glassBGR[:,:,::-1]);plt.title('Sunglass Color channels');
plt.subplot(122);plt.imshow(glassMask1,cmap='gray');plt.title('Sunglass Alpha channel');
```
```python
# Make a copy
#faceWithGlassesNaive = resized_faceImage.copy()
faceWithGlassesNaive = faceImage.copy()

# Replace the eye region with the sunglass image
faceWithGlassesNaive[156:236,174:374]=glassBGR

plt.imshow(faceWithGlassesNaive[...,::-1])
```
```python
# Make the dimensions of the mask same as the input image.
# Since Face Image is a 3-channel image, we create a 3 channel image for the mask
glassMask = cv2.merge((glassMask1,glassMask1,glassMask1))

# Make the values [0,1] since we are using arithmetic operations
glassMask = np.uint8(glassMask/255)

# Make a copy
faceWithGlassesArithmetic = faceImage.copy()

# Get the eye region from the face image
eyeROI= faceWithGlassesArithmetic[160:240,175:375]

# Use the mask to create the masked eye region
maskedEye = cv2.multiply(eyeROI,(1-  glassMask ))

# Use the mask to create the masked sunglass region
maskedGlass = cv2.multiply(glassBGR,glassMask)

# Combine the Sunglass in the Eye Region to get the augmented image
eyeRoiFinal = cv2.add(maskedEye, maskedGlass)

# Display the intermediate results
plt.figure(figsize=[20,20])
plt.subplot(131);plt.imshow(maskedEye[...,::-1]);plt.title("Masked Eye Region")
plt.subplot(132);plt.imshow(maskedGlass[...,::-1]);plt.title("Masked Sunglass Region")
plt.subplot(133);plt.imshow(eyeRoiFinal[...,::-1]);plt.title("Augmented Eye and Sunglass")
```
```python
# Replace the eye ROI with the output from the previous section
faceWithGlassesArithmetic[160:240,175:375]=eyeRoiFinal

# Display the final result
plt.figure(figsize=[20,20]);
plt.subplot(121);plt.imshow(faceImage[:,:,::-1]); plt.title("Original Image");
plt.subplot(122);plt.imshow(faceWithGlassesArithmetic[:,:,::-1]);plt.title("With Sunglasses");
```

### Output : 

<img width="495" height="582" alt="image" src="https://github.com/user-attachments/assets/ddf1765e-ef62-49fa-840b-ca42d1960e04" /></br>

<img width="807" height="393" alt="image" src="https://github.com/user-attachments/assets/94fe6039-3f4e-4b8e-832c-dce80ae9275b" /></br>

<img width="1486" height="327" alt="image" src="https://github.com/user-attachments/assets/99e63c1b-792c-472f-b822-a89ea0387b12" /></br>

<img width="501" height="573" alt="image" src="https://github.com/user-attachments/assets/ce50a9db-79dc-41de-a19e-e03e477c314f" /></br>

<img width="1498" height="219" alt="image" src="https://github.com/user-attachments/assets/c253e22d-16bb-4d23-8043-ba09af2c1c17" /></br>

<img width="1335" height="772" alt="image" src="https://github.com/user-attachments/assets/2d56e768-0b2a-4e02-8e8a-265b6e8e36d1" /></br>


### Result :
Thus, sunglasses were successfully added to the passport photo using Python and OpenCV. Image resizing, alpha masking, and arithmetic operations were used to blend the sunglasses naturally with the face image.
