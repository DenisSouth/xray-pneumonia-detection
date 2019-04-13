# xray-pneumonia-detection
xray-pneumonia-detection based on  https://github.com/llSourcell/AI_Startup_Prototype

## Please do not use this code in real projects, 
## I am not a medical doctor, this is just an image processing training

# RUN
1. open https://colab.research.google.com/
2. copy https://github.com/DenisSouth/xray-pneumonia-detection/blob/master/chest_xray_pneumonia.ipynb
3. change  runtime to GPU
4. make all steps in notebook one by one 

p.s.
I am not sure about original layers dense, i think its overfit the network
> x=Dense(1024,activation='relu')(x) 
> x=Dense(1024,activation='relu')(x) 
> x=Dense(512,activation='relu')(x)

Try to decrease it to 
> x=Dense(256,activation='relu')(x) 
> x=Dense(256,activation='relu')(x)
or some else

If you will get nice result in test data - please make pull request or issue :-)
