
lsblk

MOUNT_DIR=image-data
DEV_NAME=sdb

sudo mkdir -p /mnt/disks/$MOUNT_DIR
sudo chmod 777 /mnt/disks/$MOUNT_DIR
sudo mount -o discard,defaults /dev/$DEV_NAME /mnt/disks/$MOUNT_DIR


cd /mnt/disks/$MOUNT_DIR

sudo wget https://storage.googleapis.com/inat_data_2018_us/test2018.json.tar.gz
sudo wget https://storage.googleapis.com/inat_data_2018_us/train2018.json.tar.gz
sudo wget https://storage.googleapis.com/inat_data_2018_us/val2018.json.tar.gz

sudo wget https://storage.googleapis.com/inat_data_2018_us/test2018.tar.gz
sudo wget https://storage.googleapis.com/inat_data_2018_us/train_val2018.tar.gz

# check MD5, then tar xz
md5sum train_val2018.tar.gz #b1c6952ce38f31868cc50ea72d066cc3
md5sum test2018.tar.gz #4b71d44d73e27475eefea68886c7d1b1

tar xzf test2018.tar.gz &
tar xzf train_val2018.tar.gz &


for file in ./*.tar.gz; do
    echo $file
    sudo tar xzf $file
done


#gsutil cp -r ~/data gs://coastal-epigram-162302/inaturalist
