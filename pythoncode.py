import pandas as pd
import numpy as np
import cv2
import os
import re
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize']=(10.0,10.0)

INPUT_DIR = os.path.abspath('C:\\Users\\Maria\\Downloads\\global-wheat-detection')
TRAIN_DIR = os.path.join(INPUT_DIR, "train")

# Load and Show Training Labels
df=pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
print(df)
#function to test out seeing an image from the training set
def read_image_from_path(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def read_image_from_train_folder(image_id):
    path = os.path.join(TRAIN_DIR, image_id + ".jpg")
    return read_image_from_path(path)
sample_image_id = "b6ab77fd7"
plt.imshow(read_image_from_train_folder(sample_image_id))
_ = plt.title(sample_image_id)
plt.show()
#trying to parse the bounding boxes(strings to list of floating point values)
'''
def parse_bboxes(string_input):
    input_without_brackets = re.sub(r'\[|\]', '', string_input)
    input_as_list =np.array(input_without_brackets.split(","))
    return input_as_list.astype(np.float64)'''
#tringto get cordinates of topleft and bottom right corners
'''def xywh_to_x1y1x2y2(x,y,w,h):
    return np.array([x,y,x+w,y+h])'''


def parse_bbox_text(string_input):
    input_without_brackets = re.sub("\[|\]", "", string_input)
    input_as_list = np.array(input_without_brackets.split(","))
    return input_as_list.astype(np.float32) 

def xywh_to_x1y1x2y2(x,y,w,h):
    return np.array([x,y,x+w,y+h])
# Parse training bounding box labels
train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
bbox_series = train_df.bbox.apply(parse_bbox_text)

xywh_df = pd.DataFrame(bbox_series.to_list(), columns=["x", "y", "w", "h"])
xywh_df.reset_index(drop=True, inplace=True)

x2_df = pd.DataFrame(xywh_df.x + xywh_df.w, columns=["x2"]).reset_index(drop=True)
y2_df = pd.DataFrame(xywh_df.y + xywh_df.h, columns=["y2"]).reset_index(drop=True)

# Update training dataframe with parsed labels
train_df = pd.concat([train_df, xywh_df, x2_df, y2_df], axis=1)
train_df.head()
pd.set_option('display.max_columns', None)  # This line will ensure all columns are displayed
print(train_df.head())
#draw bounding boxes on the image
def draw_boxes_on_image(boxes,image,color=(255,0,0)):
    for box in boxes:
        cv2.rectangle(image,(int(box[0]),int(box[1])),
                      (int(box[2]),int(box[3])),
                      color,3)
    return image
#randomises them.
sample_image_id = train_df.image_id.sample().item()
sample_image = read_image_from_train_folder(sample_image_id)
sample_bounding_boxes = train_df[train_df.image_id == sample_image_id][["x", "y", "x2", "y2"]]
plt.imshow(draw_boxes_on_image(sample_bounding_boxes.to_numpy(), sample_image, color=(0, 200, 200)))
_ = plt.title(sample_image_id)
plt.show()


model = fasterrcnn_resnet50_fpn(pretrained=True)
#print(model)
#changing it to match our classes and data set
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features, num_classes=2)

# Verify the model architecture
model.roi_heads
#print(model.roi_heads)
#where to run the model so it doesnt take forever to run
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#move the images in batches to the device with an 80 / 20 split
def move_batch_to_device(images, targets):
    images= list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    return images, targets

#split data into training and validation sets
unique_image_ids = train_df['image_id'].unique()
n_validation = int(0.2 * len(unique_image_ids))
valid_ids =unique_image_ids[-n_validation:]
train_ids = unique_image_ids[:-n_validation]
validation_df = train_df[train_df['image_id'].isin(valid_ids)]
training_df = train_df[train_df['image_id'].isin(train_ids)]
print("%i training samples\n%i validation samples" % (len(training_df.image_id.unique()), len(validation_df.image_id.unique())))

#Inherit from pytorch dataset class 
#funtion to allow us to work with the pytorch dataset class which will generate batches of data and we just have to show it how to load the sample.
class WheatDataset(Dataset):
    def __init__(self, dataframe):
        super().__init__()
        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index:int):
            image_id = self.image_ids[index]
            image =read_image_from_train_folder(image_id).astype(np.float32)
            #scale image since pytorch expects images to be in the range of 0-1
            image /= 255.0
            #rearrange to match the expectations of the channels and convert it into a tensor
            image = torch.from_numpy(image).permute(2,0,1)
            records = self.df[self.df['image_id'] == image_id]
            boxes = records[['x', 'y', 'x2', 'y2']].values
            #convert to tensor of float32
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            n_boxes = boxes.shape[0]
            labels = torch.ones((n_boxes,), dtype=torch.int64)
            target = {}
            target['boxes'] = boxes
            target['labels'] = labels
            return image, target


train_dataset = WheatDataset(training_df)
valid_dataset = WheatDataset(validation_df)
#ensures everything is in the batches to make sure them all fit
def collate_fn(batch):
    return tuple(zip(*batch))
is_training_on_cpu = device == torch.device('cpu')
batch_size = 4 if is_training_on_cpu else 16
train_data_loader = DataLoader(train_dataset,
                                batch_size=batch_size,
                                shuffle=True, 
                                collate_fn=collate_fn)
valid_data_loader = DataLoader(valid_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4,
                            collate_fn=collate_fn)
'''
batch_of_images, batch_of_targets = next(iter(train_data_loader))
sample_boxes = batch_of_targets[0]['boxes'].numpy().astype(np.int32)
sample_image = batch_of_images[0].permute(1,2,0).cpu().numpy()
plt.imshow(draw_boxes_on_image(sample_boxes, sample_image,color = (0, 200, 200)))'''# Test the data loader
batch_of_images, batch_of_targets = next(iter(train_data_loader))

sample_boxes = batch_of_targets[0]['boxes'].cpu().numpy().astype(np.int32)
sample_image = batch_of_images[0].permute(1,2,0).cpu().numpy() # convert b ack from pytorch format

plt.imshow(draw_boxes_on_image(sample_boxes, sample_image, color=(0,200,200)))
plt.show()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005,momentum=0.9)
num_epochs = 1 if is_training_on_cpu else 8
model = model.to(device)
model.train()
for epoch in range(num_epochs):
    print("Epoch %i/%i " % (epoch + 1, num_epochs) )
    average_loss = 0
    for batch_id, (images, targets) in enumerate(train_data_loader):
        images, targets = move_batch_to_device(images, targets)
        loss_dict = model(images, targets)
        batch_loss = sum(loss for loss in loss_dict.values()) / len(loss_dict) 
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        loss_value= batch_loss.item()
        average_loss = average_loss + (loss_value - average_loss) / (batch_id + 1)
        print("Mini-batch: %i/%i Loss: %.4f" % ( batch_id + 1, len(train_data_loader), average_loss), end='\r')
        if batch_id % 100 == 0:
            print("Mimicats")

            print("Mini-batch: %i/%i Loss: %.4f" % ( batch_id + 1, len(train_data_loader), average_loss))
