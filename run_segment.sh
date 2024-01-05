#run_segment.sh
rm  -r /project/train/src_repo/dataset

#创建数据集相关文件夹
mkdir /project/train/src_repo/dataset
mkdir /project/train/src_repo/dataset/Annotations
mkdir /project/train/src_repo/dataset/images
mkdir /project/train/src_repo/dataset/ImageSets
mkdir /project/train/src_repo/dataset/labels
mkdir /project/train/src_repo/dataset/ImageSets/Main

find /home/data/2783/ -name "*.jpg" | xargs -i cp {} /project/train/src_repo/dataset/images
find /home/data/2783/ -name "*.png" | xargs -i cp {} /project/train/src_repo/dataset/Annotations

find /home/data/2775/ -name "*.jpg" | xargs -i cp {} /project/train/src_repo/dataset/images
find /home/data/2775/ -name "*.png" | xargs -i cp {} /project/train/src_repo/dataset/Annotations

find /home/data/2777/ -name "*.jpg" | xargs -i cp {} /project/train/src_repo/dataset/images
find /home/data/2777/ -name "*.png" | xargs -i cp {} /project/train/src_repo/dataset/Annotations

find /home/data/2779/ -name "*.jpg" | xargs -i cp {} /project/train/src_repo/dataset/images
find /home/data/2779/ -name "*.png" | xargs -i cp {} /project/train/src_repo/dataset/Annotations

find /home/data/2864/ -name "*.jpg" | xargs -i cp {} /project/train/src_repo/dataset/images
find /home/data/2864/ -name "*.png" | xargs -i cp {} /project/train/src_repo/dataset/Annotations

find /home/data/2865/ -name "*.jpg" | xargs -i cp {} /project/train/src_repo/dataset/images
find /home/data/2865/ -name "*.png" | xargs -i cp {} /project/train/src_repo/dataset/Annotations

find /home/data/2866/ -name "*.jpg" | xargs -i cp {} /project/train/src_repo/dataset/images
find /home/data/2866/ -name "*.png" | xargs -i cp {} /project/train/src_repo/dataset/Annotations

find /home/data/2869/ -name "*.jpg" | xargs -i cp {} /project/train/src_repo/dataset/images
find /home/data/2869/ -name "*.png" | xargs -i cp {} /project/train/src_repo/dataset/Annotations

find /home/data/2870/ -name "*.jpg" | xargs -i cp {} /project/train/src_repo/dataset/images
find /home/data/2870/ -name "*.png" | xargs -i cp {} /project/train/src_repo/dataset/Annotations

find /home/data/2874/ -name "*.jpg" | xargs -i cp {} /project/train/src_repo/dataset/images
find /home/data/2874/ -name "*.png" | xargs -i cp {} /project/train/src_repo/dataset/Annotations

python /project/train/src_repo/split_train_val.py --xml_path /project/train/src_repo/dataset/Annotations  --txt_path /project/train/src_repo/dataset/ImageSets/Main
cp /project/train/src_repo/png_label.py /project/train/src_repo/dataset
python /project/train/src_repo/dataset/png_label.py

yolo task=segment mode=train data=/project/train/src_repo/data_seg.yaml model=/project/train/src_repo/pretrain_model/yolov8n-seg.pt project=/project/train/models/train batch=32 epochs=50 imgsz=640 workers=16 device=0

