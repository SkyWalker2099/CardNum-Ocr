import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
datagen = ImageDataGenerator(featurewise_center=True,
							  fill_mode='nearest',
							  rescale=1./ 255,
							 featurewise_std_normalization=True,
							 zca_whitening=True,
							 horizontal_flip=True,

		)
path = os.listdir("D:/cards/KreasReshape/")
i=0
for picture in path:
	img = load_img("D:/cards/KreasReshape/"+ picture)
	x = img_to_array(img)
	x = x.reshape((1,)+x.shape)
	picturename = picture.split(".", 2)[0]
	for batch in datagen.flow(x, batch_size=1,
							  save_to_dir='D:/cards/KreasReshape', save_prefix=picturename, save_format='png'):

		i += 1
		if i>=1:
			break
	# 第五次