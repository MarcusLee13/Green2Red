import os
import shutil
import cv2
from PIL import Image

# 直接换滴度就可以
root_folder_path = "320/"

green_train_dataset_path = "trainA/"
red_train_dataset_path = "trainB/"
green_test_dataset_path = "testA/"
red_test_dataset_path = "testB/"

train_green_path = "yingguangseg/train/A/"
train_red_path = "yingguangseg/train/B/"
test_green_path = "yingguangseg/test/A/"
test_red_path = "yingguangseg/test/B/"


def rm_Monkey_ACL(root_folder, green_dataset_path, red_dataset_path):
    file_list = os.listdir(root_folder)
    for n in range(len(file_list)):
        print("This is in" + file_list[n] + " folder")
        now_folder_path = root_folder + file_list[n] + "/"
        img_list = os.listdir(now_folder_path)
        for i in range(len(img_list)):
            if "Monkey" in img_list[i] or "CAL" in img_list[i]:
                # print(img_list[i])
                # print(root_folder_path + "")
                rm_path = now_folder_path + img_list[i]
                print('remove'+rm_path)
                os.remove(rm_path)
            elif "RC" in img_list[i]:
                mv_path = now_folder_path + img_list[i]
                print(mv_path)
                shutil.copy(mv_path, red_dataset_path)
            else:
                mv_path = now_folder_path + img_list[i]
                print(mv_path)
                shutil.copy(mv_path, green_dataset_path)


def seg_quator_img(train_A_path, train_B_path, test_A_path, test_B_path):
    print(train_A_path, train_B_path, test_A_path, test_B_path)
    img_dir = [train_A_path, train_B_path, test_A_path, test_B_path]
    for dirs in img_dir:
        img_name_list = os.listdir(dirs)
        for imgs in img_name_list:
            img_PCR = dirs + imgs
            im = cv2.imread(img_PCR)
            for i in range(4):
                for n in range(4):
                    # (h,w)
                    # 竖着切的属于是
                    ims = im[(445 * i):(445 * (i + 1)), (594 * n):(594 * (n + 1))]
                    save_name = imgs.split(".")[0] + str(i) + "_" + str(n)
                    save_img_path = os.path.join(dirs, save_name + '.jpg')
                    ims = cv2.resize(ims, (596, 448))
                    cv2.imwrite(save_img_path, ims)
                    print(save_img_path)
        # rm_large_img(dirs)

def rm_large_img(rm_dir):
    imgs = os.listdir(rm_dir)
    for img_name in imgs:
        img_pth = os.path.join(rm_dir,img_name)
        im = cv2.imread(img_pth)
        # print(im.shape)
        if im.shape[0] > 448:
            os.remove(img_pth)
            print(img_pth+" is removed")

if __name__ == '__main__':
    # rm_Monkey_ACL(root_folder_path, green_test_dataset_path, red_test_dataset_path)
    # seg_quator_img(train_A_path=train_green_path, train_B_path=train_red_path, test_A_path=test_green_path, test_B_path=test_red_path)
    '''
    正确切割方法
    img = Image.open("testimgs/test/B/20210906A1_42_100_G_HEp-2 Cells, Human_0_383969___RC.jpg")
    print(img.size)
    w, h= img.size
    cropped = []
    cropped.append(img.crop((0,0,w//2,h//2)))
    cropped.append(img.crop((w//2,0,w,h//2)))
    cropped.append(img.crop((0,h//2,w//2,h)))
    cropped.append(img.crop((w//2,h//2,w,h)))
    for i in range(4):
        print(cropped[i].size)
        cropped[i].save("./%d.jpg"%(i))
    '''
    # rm_Monkey_ACL(root_folder_path, green_test_dataset_path, red_test_dataset_path)
    # seg_quator_img(train_A_path=train_green_path,train_B_path=train_red_path,test_A_path=test_green_path,test_B_path=test_red_path)
    seg_img_dir = [train_green_path, train_red_path, test_green_path, test_red_path]
    for folders in seg_img_dir:
        rm_large_img(folders)