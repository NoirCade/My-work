from utils import *
from PIL import Image

all_path = image_file_list('./data/mountain_dew')

for image_path in all_path:
    label = image_path.split('\\')[0].split('/')[2]
    file_name = image_path.split('\\')[1].split('.')[0]
    print(label, file_name)
    image = Image.open(image_path)
    image = expand2square(image, (0, 0, 0)).resize((224, 224))
    save_folder = './data/resized_' + label + '/'
    os.makedirs(save_folder, exist_ok=True)
    image.save(save_folder + file_name + '.png', 'png')
