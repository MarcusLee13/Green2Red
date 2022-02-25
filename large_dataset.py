import os
from PIL import Image
import os
import shutil
import cv2

didu_path = "3200/"
green_train_dataset_path = "trainA/"
red_train_dataset_path = "trainB/"
save_dir = "large/train/"
A_list = os.listdir(green_train_dataset_path)
B_list = os.listdir(red_train_dataset_path)

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
                print('remove' + rm_path)
                os.remove(rm_path)
            elif "RC" in img_list[i]:
                mv_path = now_folder_path + img_list[i]
                print(mv_path)
                shutil.copy(mv_path, red_dataset_path)
            else:
                mv_path = now_folder_path + img_list[i]
                print(mv_path)
                shutil.copy(mv_path, green_dataset_path)


def img_to_quator(train_A, train_B):
    img_dir=[train_A,train_B]
    for dirs in img_dir:
        img_name_list = os.listdir(dirs)
        for imgs in img_name_list:
            img_PCR = dirs + imgs
            im = cv2.imread(img_PCR)
            for i in range(2):
                for n in range(2):
                    ims = im[(890 * i):(890 * (i + 1)), (1188 * n):(1188 * (n + 1))]
                    save_name = imgs.split(".")[0] + str(i) + "_" + str(n)
                    save_img_path = os.path.join(dirs, save_name + '.jpg')
                    ims = cv2.resize(ims, (1024, 1024))
                    cv2.imwrite(save_img_path, ims)
                    print(save_img_path)

def rm_large_img(rm_dir):
    imgs = os.listdir(rm_dir)
    for img_name in imgs:
        img_pth = os.path.join(rm_dir,img_name)
        im = cv2.imread(img_pth)
        # print(im.shape)
        if im.shape[0] > 1024:
            os.remove(img_pth)
            print(img_pth+" is removed")

def pic_joint(path1, path2, save_path, flag='horizontal'):
    """
    :param path1: path
    :param path2: path
    :param save_path: path
    :param flag: horizontal or vertical
    :return:
    """
    img1,img2 = Image.open(path1), Image.open(path2)
    # 自定义设置宽高(读者自行修改要拼接的图片分辨率)
    img1 = img1.resize((1024, 1024), Image.ANTIALIAS)
    img2 = img2.resize((1024, 1024), Image.ANTIALIAS)
    size1, size2 = img1.size, img2.size
    if flag == 'horizontal':
        #创建Image对象(模式，大小)，横向宽度就是两张图片相加，高度不变
        joint = Image.new('RGB', (size1[0] + size2[0], size1[1]))
        loc1, loc2 = (0, 0), (size1[0], 0)
        #按位置拼接，二元组
        joint.paste(img1, loc1)
        joint.paste(img2, loc2)
        #读者自定义保存的路径
        joint.save(save_path)
    elif flag == 'vertical':
        joint = Image.new('RGB', (size1[0], size1[1] + size2[1]))
        loc1, loc2 = (0, 0), (0, size1[1])
        joint.paste(img1, loc1)
        joint.paste(img2, loc2)
        joint.save(save_path)

if __name__ == '__main__':
    # 先全都存储在trainA和trainB，然后切割之后删除trainA和trainB中大于标准切割图片
    # rm_Monkey_ACL(root_folder=didu_path, green_dataset_path=green_train_dataset_path,
    #               red_dataset_path=red_train_dataset_path)
    # img_to_quator(train_A=green_train_dataset_path,train_B=red_train_dataset_path)
    # rm_large_img(green_train_dataset_path)
    # rm_large_img(red_train_dataset_path)
    for i in range(540):
        im_name = str(i) + ".jpg"
        save_name = os.path.join(save_dir, im_name)
        A = os.path.join(green_train_dataset_path, A_list[i])
        B = os.path.join(red_train_dataset_path, B_list[i])
        pic_joint(B, A, save_name)
