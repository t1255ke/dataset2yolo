import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm
def show_img(img):

    plt.imshow(img)
    plt.show()
# 转换kitti数据标签labels
def build_dir(root):
    if not os.path.exists(root):
        os.makedirs(root)
    return root
def read_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    labels, boxes = [], []
    for line in lines:
        data = line.split(" ")
        label = data[0]
        xmin, ymin, xmax, ymax = float(data[4]), float(data[5]), float(data[6]), float(data[7])
        box2d = np.array([xmin, ymin, xmax, ymax])
        labels.append(label)
        boxes.append(box2d)

    return labels, boxes
def save_txt(txt_info, out_dir):
    txt_name = txt_info['txt_name']
    info = txt_info['info']
    out_file = open(os.path.join(out_dir, txt_name), 'w')  # 转换后的txt文件存放路径
    for a in info:
        cls_id = a[0]
        box = a[1:]
        out_file.write(str(cls_id) + " " + " ".join([str(b) for b in box]) + '\n')
def save_image_txt(yolo_txt_info,save_path,mode_name):
    # 保存图片与txt
    images_save_dir = build_dir(os.path.join(save_path, 'images', mode_name))
    labels_save_dir = build_dir(os.path.join(save_path, 'labels', mode_name))
    for num, info in tqdm(enumerate(yolo_txt_info)):
        save_txt(info, labels_save_dir)
        img_path = info['img_path']
        img_name = info['img_name']
        shutil.copy(img_path,os.path.join(images_save_dir, img_name))
    print("数据集{}:\t{}张".format(mode_name,num+1))
def kitti2yolo_main(path, save_path, mode_name='train', save_draw_img=False,split_val_ratio=None):

    batch_labels,batch_boxes,batch_images,batch_images_path,batch_images_shape,batch_images_name,categories=[],[],[],[],[],[],[]
    # 从kitti数据中获取相应内容
    image_names = [name for name in os.listdir(os.path.join(path, 'image_2')) if name[-3:] == 'png']
    for k,name in tqdm(enumerate(image_names)):
        # if k>10:break
        image_2_path = os.path.join(path, 'image_2', name)  # 获得图像
        label_2_path = os.path.join(path, 'label_2', name[:-4] + '.txt')  # 获得标签lebl
        img = cv2.imread(image_2_path) # 读取图像
        if img is None:
            print('图像存在问题：',image_2_path)
            continue
        labels, boxes = read_label(label_2_path)  # 处理标签
        # for 3d bbox
        # TODO: change the color of boxes
        boxes_tmp, labels_tmp = [], []  # label,x1,y1,x2,y2
        for i, cat in enumerate(labels):
            # 画2d坐标
            if cat == "DontCare":continue
            box = boxes[i]

            labels_tmp.append(cat)
            boxes_tmp.append(boxes[i])
            if cat not in categories:categories.append(cat)
            if save_draw_img:
                cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2, )
                cv2.putText(img, cat, (int(box[0]), int(box[1]) if int(box[1])>10 else int(box[1]+10) ), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0), 2)
        batch_images.append(img)
        batch_labels.append(labels_tmp)
        batch_boxes.append(boxes_tmp)
        batch_images_path.append(image_2_path)
        batch_images_shape.append(img.shape)
        batch_images_name.append(name)
    categories=list(np.sort(categories))
    # 转换成yolo格式
    yolo_txt_info = []
    for i, labels in tqdm(enumerate(batch_labels)):
        img_path = batch_images_path[i]
        if labels ==[]:
            print(img_path)
            continue
        bboxes = batch_boxes[i]
        # img_path = batch_images_path[i]
        H, W, C = batch_images_shape[i]
        img_name = batch_images_name[i]
        text_name = img_name[:-3] + "txt"
        txt_info = {'info': [], 'txt_name': text_name,'img_name':img_name, 'img_path': img_path}
        for j, cat in enumerate(labels):
            cat_id = list(categories).index(cat)
            x1, y1, x2, y2 = bboxes[j]
            x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
            dw = 1.0 / W
            dh = 1.0 / H
            x = x * dw
            w = w * dw
            y = y * dh
            h = h * dh
            txt_info['info'].append([cat_id, x, y, w, h])
        yolo_txt_info.append(txt_info)

    # 保存图片与txt
    if split_val_ratio:
        Num = len(yolo_txt_info)
        split_idx = int((1-split_val_ratio)*Num)
        yolo_val_info = yolo_txt_info[split_idx:]
        yolo_txt_info = yolo_txt_info[:split_idx]
        save_image_txt(yolo_val_info, save_path, 'val')
    save_image_txt(yolo_txt_info, save_path, mode_name)
    print('数据存放路径：',save_path)
    print('类别标签：', categories)

    # 保存画框和标签图片
    if save_draw_img:
        save_draw_dir = build_dir(os.path.join(save_path, 'draw_info_on_images'))
        for i, name in enumerate(batch_images_name):
            save_draw_path = os.path.join(save_draw_dir,name)
            cv2.imwrite(save_draw_path,batch_images[i])


    return yolo_txt_info,categories

if __name__ == '__main__':
    path = r'D:\t1255\Document\NTUT\inpainting\dataset\data_semantics\training'
    save_path = r'D:\t1255\Document\NTUT\inpainting\dataset\data_semantics\kitti_data'
    kitti2yolo_main(path, save_path, mode_name='train', save_draw_img=True, split_val_ratio=0.2)

