import os
from sklearn.model_selection import train_test_split

def preprocess(root_dir, output_dir):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            os.mkdir(os.path.join(output_dir, 'train'))
            os.mkdir(os.path.join(output_dir, 'test'))

        # Split train/val/test sets
        for file in os.listdir(root_dir):
            file_path = os.path.join(root_dir, file)               #ex:root_dir = (/media/brian/Brian/2023/splited_pcap/), file = SSH
            video_files = [name for name in os.listdir(file_path)]      #ex: video_file = [SSH_1, SSH2, ...](all attack folder)

            train, test = train_test_split(video_files, test_size=0.2, random_state=42)

            train_dir = os.path.join(output_dir, 'train', file)
            test_dir = os.path.join(output_dir, 'test', file)

            if not os.path.exists(train_dir):
                os.makedirs(train_dir)
            if not os.path.exists(test_dir):
                os.makedirs(test_dir)

            print("train:", train)  
            print("test:", test)
            for att_folder in train:
                command = 'cp -r '+ file_path + "/" + att_folder + " " + train_dir
                print(command)
                os.system(command)

            for att_folder in test:
                command = 'cp -r '+ file_path + "/" + att_folder + " " + test_dir
                os.system(command)

        print('Preprocessing finished.')

if __name__ == "__main__":
    preprocess("/SSD/p76111262/CIC-IDS-2018-v1/Other/", "/SSD/p76111262/CIC-IDS2018-v1/Other/")