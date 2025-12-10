#!/usr/bin/bash
WORKSPACE="/home/tsyo/work/gpt2-zhtw/workspace/"
UTIL_PATH="/home/tsyo/work/daily_poem/"

cd $WORKSPACE

# Fetch daily data
python $UTIL_PATH/dp1_retrieve_prossed_data.py

for i in {1..5}
  do
# Generate poem
    python $UTIL_PATH/dp2_generate_poem_from_data.py -c config_gulong.json
    PTITLE=`head -n 1 poem.txt`
# Generate image
    python $UTIL_PATH/dp3_generate_images_from_poem.py
# Send to blogger
    mutt -s "$PTITLE" -a poem.jpg -- tingshuo.yo.zhyx@blogger.com < poem.txt
  done