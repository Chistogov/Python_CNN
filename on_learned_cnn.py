from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Каталог с данными для тестирования
test_dir = 'test'
# Размеры изображения
img_width, img_height = 150, 150
# Размер мини-выборки
batch_size = 16
# Количество изображений для тестирования
nb_test_samples = 100

print("Загружаю сеть из файлов")
# Загружаем данные об архитектуре сети из файла json
json_file = open("mnist_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
# Создаем модель на основе загруженных данных
loaded_model = model_from_json(loaded_model_json)
# Загружаем веса в модель
loaded_model.load_weights("mnist_model.h5")
print("Загрузка сети завершена")

# Компилируем модель
loaded_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# datagen = ImageDataGenerator(rescale=1. / 255)

# test_generator = datagen.flow_from_directory(
#     test_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode='binary')

# scores = loaded_model.evaluate_generator(test_generator, nb_test_samples // batch_size)

# image = cv2.imread("D:/PythonProjects/untitled1/train/dogs/dog.6053.jpg")
# image = cv2.resize(image, (150, 150))
# image2 = img_to_array(image)
# image2 = np.expand_dims(image2, axis=0)
# cat = loaded_model.predict_classes(image2)
#
# # print(rt)
# classname = cat[0]
# print ("Class: ",classname)
#
# plt.imshow(image)
# plt.title(classname)
# plt.show()
c = 0
d = 0
for i in range(1500):
    i2=i+10700
    img = image.load_img(path="D:/PythonProjects/untitled1/test/cats/cat."+str(i2)+".jpg",target_size=(150,150,3))
    img = image.img_to_array(img)
    test_img = np.expand_dims(img, axis=0)
    img_class = loaded_model.predict_classes(test_img)
    print (img_class)
    classname = img_class[0]
    if(classname < 0.5):
        print ("Cat")
        c+=1
    if (classname > 0.5):
        print ("Dog")
        d+=1

print ("Cat: "+str(c) + "/1500")
print ("Dog: "+str(d) + "/1500")
plt.imshow(img)
plt.title(classname)
plt.show()
# Train imgs
# Cats
# Cat: 1514/2000
# Dog: 486/2000
# Dogs
# Cat: 24/2000
# Dog: 1976/2000

# Test imgs
# Cats
# Cat: 906/1500
# Dog: 594/1500
# Dogs
# Cat: 103/1500
# Dog: 1397/1500