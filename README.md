# Healthcare
Automatic Diabetic Retinopathy Detection using pretrained models

1.Use predict.py to test an image.

2.link for ResNet model trained with 30 epochs when only classifier is trained
https://drive.google.com/file/d/14gKzlEUiipWXxfL94CQNftFOmb3t31Gf/view?usp=sharing

3.link for ResNet model trained with 30 epochs when following layers are added:
https://drive.google.com/file/d/1gvKICsx8S5BEWLbek_gKJAWdbQELwTED/view?usp=sharing

# Fine tune the resnet 50
#image_input = Input(shape=(224, 224, 3))
model = ResNet50(weights='imagenet',include_top=False)
model.summary()
last_layer = model.output
# add a global spatial average pooling layer
x = GlobalAveragePooling2D()(last_layer)
# add fully-connected & dropout layers
x = Dense(512, activation='relu',name='fc-1')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu',name='fc-2')(x)
x = Dropout(0.5)(x)
# a softmax layer for 4 classes
out = Dense(num_classes, activation='softmax',name='output_layer')(x)
# this is the model we will train
custom_resnet_model2 = Model(inputs=model.input, outputs=out)
custom_resnet_model2.summary()


