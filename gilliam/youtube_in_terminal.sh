#!/bin/sh

# brew update
# brew install ffmpeg youtube-dl jp2a

VIDEO_URL="https://www.youtube.com/watch?v="
VIDEO_ID="-rutX0I6NxU"
FRAME_RATE=24

while getopts "v:f:" opt; do
  case ${opt} in
    v )
      VIDEO_ID=$OPTARG
      ;;
    f )
      FRAME_RATE=$OPTARG
      ;;
    \? ) echo "Usage: youtube_in_terminal [-v youtube video identifier] [-f FPS frame rate]"
      ;;
  esac
done

WAIT_SECS=$(echo $FRAME_RATE^-1 | bc -l)

# Pipe youtube video through ffmpeg to make thumbnails
youtube-dl --no-playlist --geo-bypass \
    -q "${VIDEO_URL}${VIDEO_ID}" -o - | \
ffmpeg -hide_banner -loglevel panic -i pipe:0 \
    -framerate $FRAME_RATE -vsync 0 \
    -filter:v "gblur=sigma=1.0, normalize=blackpt=black:whitept=white, edgedetect=mode=colormix:low=0.1:high=0.4, normalize=blackpt=black:whitept=white" -pix_fmt yuv420p \
    /tmp/thumb_${VIDEO_ID}_%05d.jpg

for image in $(ls -1 /tmp/thumb_${VIDEO_ID}_*.jpg)
do
    jp2a --clear --term-fit $image;
    rm $image;
    sleep $WAIT_SECS;
done
rm /tmp/thumb_${VIDEO_ID}_*.jpg || true;
