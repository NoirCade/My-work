import os
from PIL import Image
import torch
import time
import copy

IMG_FORMATS = ['.jpg', '.png', '.JPG', '.PNG']


def image_file_list(image_folder_path):
    # 폴더에서 파일 서치하는 함수

    all_root = []
    for (path, dir, files) in os.walk(image_folder_path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext in IMG_FORMATS:
                root = os.path.join(path, filename)
                all_root.append(root)
            else:
                print('no image found...')
                continue

    return all_root


def expand2square(pil_image, background_color):
    width, height = pil_image.size
    if width == height:
        return pil_image
    elif width > height:
        result = Image.new(pil_image.mode, (width, width), background_color)
        result.paste(pil_image, (0, (width-height) // 2))
        return result
    else:
        result = Image.new(pil_image.mode, (height, height), background_color)
        result.paste(pil_image, ((height-width) // 2, 0))
        return result


def train(model, criterion, train_loader, val_loader, optimizer, num_epochs, device):
    total = 0
    best_loss = 9999
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    train_loss_ls = []
    val_loss_ls = []
    train_batch = 0
    train_loss = 0
    train_acc = 0
    tLoss = 0
    tAcc = 0
    vltemp = 0

    for epoch in range(num_epochs) :
        print(f"Epoch {epoch} / {num_epochs}")
        print("-"*10)

        for index, (images, labels) in enumerate(train_loader) :
            image, label = images.to(device), labels.to(device)
            output = model(image)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, argmax = torch.max(output, 1)
            acc = (label == argmax).float().mean()
            total += label.size(0)

            train_loss = train_loss + ((1 / (index + 1)) * (loss.data - train_loss))
            train_acc = train_acc + ((1 / (index + 1)) * (acc - train_acc))
            tLoss += train_loss.item()
            tAcc += train_acc.item()
            train_batch += 1

            if (index + 1) % 10 == 0 :
                print("Epoch [{}/{}], Step [{}/{}], Loss {:.4f}, Acc {:.2f}".format(
                    epoch + 1, num_epochs, index+1, len(train_loader), loss.item(),
                    acc.item() * 100
                ))
        avrg_loss, val_acc, vltemp = validation(epoch, model, val_loader, criterion, device)

        train_loss_ls.append(tLoss/train_batch)
        val_loss_ls.append(vltemp)

        if avrg_loss < best_loss :
            best_loss = avrg_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            save_model(model, save_dir='./models', file_name=('mobilenetv3_' + str(epoch+1) + '.pt'))
            print('Current model saved')

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(
        time_elapsed // 60, time_elapsed % 6
    ))
    model.load_state_dict(best_model_wts)

    return train_loss_ls, val_loss_ls


def validation(epoch, model, val_loader, criterion, device) :
    print("Start validation # {}" .format(epoch+1))

    model.eval()
    with torch.no_grad() :
        total = 0
        correct = 0
        total_loss = 0
        cnt = 0
        batch_loss = 0
        val_batch = 0
        val_loss = 0
        vLoss = 0

        for i, (images, labels) in enumerate(val_loader) :
            image, label = images.to(device), labels.to(device)
            output = model(image)
            loss = criterion(output, label)
            batch_loss += loss.item()

            val_loss = val_loss + ((1 / (i + 1)) * (loss.data - val_loss))
            vLoss += val_loss.item()
            val_batch += 1

            total += image.size(0)
            _, argmax = torch.max(output, 1)
            correct += (label == argmax).sum().item()
            total_loss += loss
            cnt += 1

    avrg_loss = total_loss / cnt
    val_acc = (correct / total * 100)
    print("Validation #{} Acc : {:.2f}% Average Loss : {:.4f}".format(
        epoch + 1,
        val_acc,
        avrg_loss
    ))

    return avrg_loss, val_acc, vLoss/val_batch


def save_model(model, save_dir, file_name):
    output_path = os.path.join(save_dir, file_name)

    torch.save(model.state_dict(), output_path)


def test(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            image, label = images.to(device), labels.to(device)
            output = model(image)
            _, argmax = torch.max(output, 1)
            total += image.size(0)
            correct += (label == argmax).sum().item()

        acc = (correct / total) * 100
        print('Acc >>> {:.2f}%'.format(acc))
    model.train()
