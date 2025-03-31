# processes of knee flexion / extension angle 
pick up hip, knee, and ankle points 

filter the data with Butterworh low pass 4th zero lag method
cutoff frequency = 6Hz

event setting
event 1 = the mid hip point lower 5cm than the standing posture 
event 2 = the mid hip point back to 5cm after the event 1 

calculate knee angle - relative angles (knee flexion / extension angle)
    -> only do left knee angle

fix the gimbal lock problem (if needed)

find the maximum knee flexion angle 
-------------------------------------------
# main
1. apply the processes to both opencap and tracker data 
2. compare the data between trials and systems 
3. run ICC and rms to compare the data 
-------------------------------------------
# bouns - find linear kinematics from opencap
1. export the trc files from the opencap website
2. pick the points 

# processes the data
1. interpolation - using cubic method
2. filter data - using Butterworth 4th zero lag low pass method 
    -> cutoff frequency = 6Hz
3. time derivate 
    using function of time_d -> to get velocity 
    using function of time_dd-> to get acceleration
