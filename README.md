
# **Ego Vehicle Stop Detection**

### Objective
The objective is to predict when the ego vehicle has stopped based on the visual input​ .The
python code STOP.py prints “STOP” at the top right corner of the image whenever the ego vehicle
comes to a stop.

#### Algorithm

(1) When the ego vehicle moves, the background moves and when the ego vehicle stops,
the background stays rest. So we track feature points which are in the sky (top part of
the image ) and also on the road (bottom part of the image).

(2) If the number of points, for which the length of feature points moved from one frame to
the next frame crossing the threshold is above a certain number, then we say that the
ego vehicle has moved.

(3) The tracking algorithm used here is KLT Tracker.

#### Dependencies & my environment


* Python2.7
* OpenCV 3.0, Numpy

#### How to compile and run the code

(1) Run the following script
```sh
python STOP.py -v input_videos/1.mp4
```
(2) This script will take video from ​ input_videos​ folder, processes it and saves the output
in the ​ output_videos​ folder in the name ​ result.mp4. ​ The results which are already present in
the ​ output_videos​ folder is speeded up at a certain rate.
