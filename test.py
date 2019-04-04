import os
from my_utils import utils_paths
path = './dataset/panda/panda_00067.jpg'
l = path.split(os.path.sep)

# path.split(os.path.sep)[-2]
imagePaths = sorted(list(utils_paths.list_images('./dataset/panda/')))

delfile = []

for imagepath in imagePaths:
    if imagepath.__len__() > 43:
        os.remove(imagepath)
        delfile.append(imagepath)

        # print(imagepath)
        # break
print(delfile.__len__())
# print(delfile[:30])
# rm = './dataset/panda/panda_00024_20190402203848.jpg'
# # os.remove(rm)