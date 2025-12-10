ECHO OFF
cd C:\Users\tsyo\Dropbox\work\workspace\GPT\gpt2-zhtw\workspace
python ../utils/dp1_retrieve_prossed_data.py
python ../utils/dp2_generate_poem_from_data.py -c config_gulong.json
python ../utils/dp3_generate_images_from_poem.py
PAUSE