# Grenade_AK

This impliments a Convolutional Neural Network to classify images of grenades and AK 47!!! I have used transfer learning to modify a VGG19
model. 
I have used keras 2.0 with a tensorflow backend. Use keras_tl.py if you want to retrain your model on some other classes, you will need 
to modify
my code a little. You can use my currently trained model as a web app by using server.py.


For a demo just run sever.py, and then curl an image to /predict endpoint, below is a sample code:

> curl -F "file=@grenade_test.jpg" http://127.0.0.1:5000/predict
