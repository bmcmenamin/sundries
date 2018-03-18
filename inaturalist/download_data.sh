
#lsblk

DEV_NAME=sdb
MOUNT_DIR=image-data

sudo mkfs.ext4 -m 0 -F -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/$DEV_NAME
sudo mkdir -p /mnt/disks/$MOUNT_DIR
sudo mount -o discard,defaults /dev/$DEV_NAME /mnt/disks/$MOUNT_DIR

sudo chmod 777 /mnt/disks/$MOUNT_DIR
cd /mnt/disks/$MOUNT_DIR

curl https://storage.googleapis.com/inat_data_2018_us/test2018.json.tar.gz | tar xz
curl https://storage.googleapis.com/inat_data_2018_us/train2018.json.tar.gz | tar xz
curl https://storage.googleapis.com/inat_data_2018_us/val2018.json.tar.gz | tar xz

curl https://storage.googleapis.com/inat_data_2018_us/test2018.tar.gz | tar xz
curl https://storage.googleapis.com/inat_data_2018_us/train_val2018.tar.gz | tar xz

#gsutil cp -r ~/data gs://coastal-epigram-162302/inaturalist
