#run_detect.sh
rm  -r /project/train/src_repo/dataset
#创建数据集相关文件夹
mkdir /project/train/src_repo/dataset
mkdir /project/train/src_repo/dataset/Annotations
mkdir /project/train/src_repo/dataset/images
mkdir /project/train/src_repo/dataset/ImageSets
mkdir /project/train/src_repo/dataset/labels
mkdir /project/train/src_repo/dataset/ImageSets/Main

find /home/data/2783/ -name "*.jpg" | xargs -i cp {} /project/train/src_repo/dataset/images
find /home/data/2783/ -name "*.xml" | xargs -i cp {} /project/train/src_repo/dataset/Annotations

find /home/data/2775/ -name "*.jpg" | xargs -i cp {} /project/train/src_repo/dataset/images
find /home/data/2775/ -name "*.xml" | xargs -i cp {} /project/train/src_repo/dataset/Annotations

find /home/data/2777/ -name "*.jpg" | xargs -i cp {} /project/train/src_repo/dataset/images
find /home/data/2777/ -name "*.xml" | xargs -i cp {} /project/train/src_repo/dataset/Annotations

find /home/data/2779/ -name "*.jpg" | xargs -i cp {} /project/train/src_repo/dataset/images
find /home/data/2779/ -name "*.xml" | xargs -i cp {} /project/train/src_repo/dataset/Annotations

find /home/data/2864/ -name "*.jpg" | xargs -i cp {} /project/train/src_repo/dataset/images
find /home/data/2864/ -name "*.xml" | xargs -i cp {} /project/train/src_repo/dataset/Annotations

find /home/data/2865/ -name "*.jpg" | xargs -i cp {} /project/train/src_repo/dataset/images
find /home/data/2865/ -name "*.xml" | xargs -i cp {} /project/train/src_repo/dataset/Annotations

find /home/data/2866/ -name "*.jpg" | xargs -i cp {} /project/train/src_repo/dataset/images
find /home/data/2866/ -name "*.xml" | xargs -i cp {} /project/train/src_repo/dataset/Annotations

find /home/data/2869/ -name "*.jpg" | xargs -i cp {} /project/train/src_repo/dataset/images
find /home/data/2869/ -name "*.xml" | xargs -i cp {} /project/train/src_repo/dataset/Annotations

find /home/data/2870/ -name "*.jpg" | xargs -i cp {} /project/train/src_repo/dataset/images
find /home/data/2870/ -name "*.xml" | xargs -i cp {} /project/train/src_repo/dataset/Annotations

find /home/data/2874/ -name "*.jpg" | xargs -i cp {} /project/train/src_repo/dataset/images
find /home/data/2874/ -name "*.xml" | xargs -i cp {} /project/train/src_repo/dataset/Annotations

#执行数据集划分�?�转�?
python /project/train/src_repo/split_train_val.py --xml_path /project/train/src_repo/dataset/Annotations  --txt_path /project/train/src_repo/dataset/ImageSets/Main
cp /project/train/src_repo/voc_label.py /project/train/src_repo/dataset
python /project/train/src_repo/dataset/voc_label.py
python /project/train/src_repo/ViewCategory.py
#执行YOLO训练脚本
yolo task=detect mode=train data=/project/train/src_repo/data.yaml model=/project/train/src_repo/pretrain_model/yolov8n.pt project=/project/train/models/train batch=32 epochs=50 imgsz=640 workers=16 device=0