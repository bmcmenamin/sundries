#/bin/bash

##
## Mount persistent disk
##

lsblk

MOUNT_DIR=image-data
read -p "Name of device to mount (leave empty to skip):" DEV_NAME

if [ -n "$DEV_NAME" ]; then
    sudo mkdir -p /mnt/disks/$MOUNT_DIR
    sudo mount -o discard,defaults /dev/$DEV_NAME /mnt/disks/$MOUNT_DIR
fi
