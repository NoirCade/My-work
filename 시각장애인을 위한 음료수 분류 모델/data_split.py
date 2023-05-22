import glob
import os.path
from PIL import Image
from sklearn.model_selection import train_test_split

chilsung_image = glob.glob(os.path.join('./data/chilsung/*.png'))
coca_image = glob.glob(os.path.join('./data/coca/*.png'))
letsbe_image = glob.glob(os.path.join('./data/letsbe/*.png'))
milkis_image = glob.glob(os.path.join('./data/milkis/*.png'))
mountain_dew_image = glob.glob(os.path.join('./data/mountain_dew/*.png'))
welchs_image = glob.glob(os.path.join('./data/welchs/*.png'))

os.makedirs('./dataset', exist_ok=True)

# chilsung
chilsung_train, chilsung_temp = train_test_split(chilsung_image, test_size=0.2, random_state=77)
chilsung_val, chilsung_test = train_test_split(chilsung_temp, test_size=0.5, random_state=77)

os.makedirs('./dataset/train/chilsung/', exist_ok=True)
for i in chilsung_train:
    chilsung_train_path = './dataset/train/chilsung/'
    image_name = i.split('\\')[1].split('.')[0]
    image = Image.open(i)
    image = image.convert('RGB')
    image.save(chilsung_train_path + image_name + '.png', 'png')

os.makedirs('./dataset/val/chilsung/', exist_ok=True)
for i in chilsung_val:
    chilsung_val_path = './dataset/val/chilsung/'
    image_name = i.split('\\')[1].split('.')[0]
    image = Image.open(i)
    image = image.convert('RGB')
    image.save(chilsung_val_path + image_name + '.png', 'png')

os.makedirs('./dataset/test/chilsung/', exist_ok=True)
for i in chilsung_test:
    chilsung_test_path = './dataset/test/chilsung/'
    image_name = i.split('\\')[1].split('.')[0]
    image = Image.open(i)
    image = image.convert('RGB')
    image.save(chilsung_test_path + image_name + '.png', 'png')

print('chilsung >> ', len(chilsung_train), len(chilsung_val), len(chilsung_test))

coca_train, coca_temp = train_test_split(coca_image, test_size=0.2, random_state=77)
coca_val, coca_test = train_test_split(coca_temp, test_size=0.5, random_state=77)

os.makedirs('./dataset/train/coca/', exist_ok=True)
for i in coca_train:
    coca_train_path = './dataset/train/coca/'
    image_name = i.split('\\')[1].split('.')[0]
    image = Image.open(i)
    image = image.convert('RGB')
    image.save(coca_train_path + image_name + '.png', 'png')

os.makedirs('./dataset/val/coca/', exist_ok=True)
for i in coca_val:
    coca_val_path = './dataset/val/coca/'
    image_name = i.split('\\')[1].split('.')[0]
    image = Image.open(i)
    image = image.convert('RGB')
    image.save(coca_val_path + image_name + '.png', 'png')

os.makedirs('./dataset/test/coca/', exist_ok=True)
for i in coca_test:
    coca_test_path = './dataset/test/coca/'
    image_name = i.split('\\')[1].split('.')[0]
    image = Image.open(i)
    image = image.convert('RGB')
    image.save(coca_test_path + image_name + '.png', 'png')

print('coca >> ', len(coca_train), len(coca_val), len(coca_test))

letsbe_train, letsbe_temp = train_test_split(letsbe_image, test_size=0.2, random_state=77)
letsbe_val, letsbe_test = train_test_split(letsbe_temp, test_size=0.5, random_state=77)

os.makedirs('./dataset/train/letsbe/', exist_ok=True)
for i in letsbe_train:
    letsbe_train_path = './dataset/train/letsbe/'
    image_name = i.split('\\')[1].split('.')[0]
    image = Image.open(i)
    image = image.convert('RGB')
    image.save(letsbe_train_path + image_name + '.png', 'png')

os.makedirs('./dataset/val/letsbe/', exist_ok=True)
for i in letsbe_val:
    letsbe_val_path = './dataset/val/letsbe/'
    image_name = i.split('\\')[1].split('.')[0]
    image = Image.open(i)
    image = image.convert('RGB')
    image.save(letsbe_val_path + image_name + '.png', 'png')

os.makedirs('./dataset/test/letsbe/', exist_ok=True)
for i in letsbe_test:
    letsbe_test_path = './dataset/test/letsbe/'
    image_name = i.split('\\')[1].split('.')[0]
    image = Image.open(i)
    image = image.convert('RGB')
    image.save(letsbe_test_path + image_name + '.png', 'png')

print('letsbe >> ', len(letsbe_train), len(letsbe_val), len(letsbe_test))

milkis_train, milkis_temp = train_test_split(milkis_image, test_size=0.2, random_state=77)
milkis_val, milkis_test = train_test_split(milkis_temp, test_size=0.5, random_state=77)

os.makedirs('./dataset/train/milkis/', exist_ok=True)
for i in milkis_train:
    milkis_train_path = './dataset/train/milkis/'
    image_name = i.split('\\')[1].split('.')[0]
    image = Image.open(i)
    image = image.convert('RGB')
    image.save(milkis_train_path + image_name + '.png', 'png')

os.makedirs('./dataset/val/milkis/', exist_ok=True)
for i in milkis_val:
    milkis_val_path = './dataset/val/milkis/'
    image_name = i.split('\\')[1].split('.')[0]
    image = Image.open(i)
    image = image.convert('RGB')
    image.save(milkis_val_path + image_name + '.png', 'png')

os.makedirs('./dataset/test/milkis/', exist_ok=True)
for i in milkis_test:
    milkis_test_path = './dataset/test/milkis/'
    image_name = i.split('\\')[1].split('.')[0]
    image = Image.open(i)
    image = image.convert('RGB')
    image.save(milkis_test_path + image_name + '.png', 'png')

print('milkis >> ', len(milkis_train), len(milkis_val), len(milkis_test))

mountain_dew_train, mountain_dew_temp = train_test_split(mountain_dew_image, test_size=0.2, random_state=77)
mountain_dew_val, mountain_dew_test = train_test_split(mountain_dew_temp, test_size=0.5, random_state=77)

os.makedirs('./dataset/train/mountain_dew/', exist_ok=True)
for i in mountain_dew_train:
    mountain_dew_train_path = './dataset/train/mountain_dew/'
    image_name = i.split('\\')[1].split('.')[0]
    image = Image.open(i)
    image = image.convert('RGB')
    image.save(mountain_dew_train_path + image_name + '.png', 'png')

os.makedirs('./dataset/val/mountain_dew/', exist_ok=True)
for i in mountain_dew_val:
    mountain_dew_val_path = './dataset/val/mountain_dew/'
    image_name = i.split('\\')[1].split('.')[0]
    image = Image.open(i)
    image = image.convert('RGB')
    image.save(mountain_dew_val_path + image_name + '.png', 'png')

os.makedirs('./dataset/test/mountain_dew/', exist_ok=True)
for i in mountain_dew_test:
    mountain_dew_test_path = './dataset/test/mountain_dew/'
    image_name = i.split('\\')[1].split('.')[0]
    image = Image.open(i)
    image = image.convert('RGB')
    image.save(mountain_dew_test_path + image_name + '.png', 'png')

print('mountain_dew >> ', len(mountain_dew_train), len(mountain_dew_val), len(mountain_dew_test))

welchs_train, welchs_temp = train_test_split(welchs_image, test_size=0.2, random_state=77)
welchs_val, welchs_test = train_test_split(welchs_temp, test_size=0.5, random_state=77)

os.makedirs('./dataset/train/welchs/', exist_ok=True)
for i in welchs_train:
    welchs_train_path = './dataset/train/welchs/'
    image_name = i.split('\\')[1].split('.')[0]
    image = Image.open(i)
    image = image.convert('RGB')
    image.save(welchs_train_path + image_name + '.png', 'png')

os.makedirs('./dataset/val/welchs/', exist_ok=True)
for i in welchs_val:
    welchs_val_path = './dataset/val/welchs/'
    image_name = i.split('\\')[1].split('.')[0]
    image = Image.open(i)
    image = image.convert('RGB')
    image.save(welchs_val_path + image_name + '.png', 'png')

os.makedirs('./dataset/test/welchs/', exist_ok=True)
for i in welchs_test:
    welchs_test_path = './dataset/test/welchs/'
    image_name = i.split('\\')[1].split('.')[0]
    image = Image.open(i)
    image = image.convert('RGB')
    image.save(welchs_test_path + image_name + '.png', 'png')

print('welchs >> ', len(welchs_train), len(welchs_val), len(welchs_test))
