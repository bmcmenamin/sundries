
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


# Preprocess original data

find /mnt/disks/image-data/ -name '*.jpg' -exec sudo mogrify -resize 600x600 {} \; -exec jpeginfo {} \; | grep -iE 'exif|error|corrupt' >> /mnt/disks/image-data/exif_and_error_files.txt &

cat exif_and_error_files.txt | grep -iE EXIF  | awk '{print $1}' >> /mnt/disks/image-data/exif_files.txt

cat exif_and_error_files.txt | grep -iv 'EXIF'  | awk '{print $1}' >> /mnt/disks/image-data/error_files.txt


while read f; do
    echo $f
    sudo convert -format bmp $f $f.bmp
    sudo convert -format jpg $f.bmp $f
    sudo rm -f $f.bmp
done < /mnt/disks/image-data/exif_files.txt

mkdir test2018_bad
mv /mnt/disks/image-data/test2018/ba8376a212476203313bbfadfbe39d62.jpg test2018_bad/
mv /mnt/disks/image-data/test2018/ba448fcfdf84e5e89d77402ecc5fa3ce.jpg test2018_bad/


tar -xzvf test2018.tar.gz test2018/ba8376a212476203313bbfadfbe39d62.jpg
