import csv
import os
from torch.utils.data import Dataset
import cv2

class Features(Dataset):
    def __init__(self, path_to_csv):
        path_to_csv = os.path.join(path_to_csv, 'driving_log.csv')
        self.csv_data = self.load_csv_file(path_to_csv)

    def load_csv_file(self, path_to_csv):
        #path_to_csv = "../images_for_training_jungle/driving_log.csv"
        data = []
        with open(path_to_csv, 'r') as csvfile:
            data_reader = csv.reader(csvfile, delimiter=',')
            for row in data_reader:
                data.append(row)

        return data

    def get_csv_data(self):
        return self.csv_data

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self,i):
        data_entry = self.csv_data[i]
        img_center = cv2.imread(data_entry[0])
        img_left = cv2.imread(data_entry[1])
        img_right = cv2.imread(data_entry[2])

        to_return = {
            'img_center': img_center,
            'img_left': img_left,
            'img_right': img_right,
            'steering_angle': data_entry[3],
            'throttle': data_entry[4],
            'brake': data_entry[5],
            'speed': data_entry[6]
        }

        return to_return

if __name__ == '__main__':
    #test
    dataset = Features()
    print(dataset[0]['img'].shape)
