## content
## Project Description
The Lucas-Kanade Flow tracking method is implemented, and more than two feature points are randomly assigned on two sets of images for tracking.
## Mark feature points
Use the setMouseCallback package in OpenCV to return the coordinate value after mouse interaction, record the marked coordinate point, and present it in the form of a blue dot, it is shown below:
   <p float="left">
     <img src="data/Cup_bluepoint.jpg" width=400/> <img src="data/Pillow_bluepoint.jpg" width=400/>
   </p>
   
## Calculate Optical Flow
Use the calcOpticalFlowPyrLK package in OpenCV to get the tracked coordinate values and display the results of each iteration in the image. The default value of iteration is 20, it is shown below:
   <p float="left">
     <img src="data/Cup_track.jpg" width=400/> <img src="data/Pillow_track.jpg" width=400/>
   </p>
