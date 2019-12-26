import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
datagen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rescale=1./ 255,
    shear_range=0.5,
    zca_whitening=False,
    zca_epsilon=1e-6,
    fill_mode='nearest',
    cval=0,
    channel_shift_range=0,
    horizontal_flip=True,
    vertical_flip=False,

        )
path = os.listdir("D:/cards/images/")
for picture in path:
    img = load_img("D:/cards/images/"+ picture)
    x = img_to_array(img)
    x = x.reshape((1,)+x.shape)

    i = 0
    picturename = picture.split(".", 2)[0]
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir='D:/cards/S2', save_prefix=picturename, save_format='png'):

        i += 1

        if i>=10:
            break
    # 第二次
