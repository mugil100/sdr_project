import os

with open('assignment.grc', 'r', encoding='utf-8') as f:
    grc_data = f.read()

# Replace paths inside candidates array string literals
grc_data = grc_data.replace("'models','saved_models','encoder_fading.keras'", "'flat_fading_results','encoder_fading.keras'")
grc_data = grc_data.replace("'models','saved_models','decoder_fading.keras'", "'flat_fading_results','decoder_fading.keras'")

# Also fix the exception message
grc_data = grc_data.replace("python train.py --k 4", "python train.py --k 8 --use_fading")

with open('assignment.grc', 'w', encoding='utf-8') as f:
    f.write(grc_data)

print("Patch applied for flat_fading_results paths.")
