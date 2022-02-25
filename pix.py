import os
from PIL import Image

train_A = 'trainA'
A_list = os.listdir(train_A)
train_B = 'trainB'
B_list = os.listdir(train_B)
save_dir = 'pix2pix/train/'

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
    for i in range(78):
        im_name = str(i)+".jpg"
        save_name = os.path.join(save_dir, im_name)
        A = os.path.join(train_A,A_list[i])
        B = os.path.join(train_B,B_list[i])
        pic_joint(B, A, save_name)