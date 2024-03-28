class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'CIC-IDS2018':
            # folder that contains class labels
            root_dir = '/SSD/p76111262/CIC-IDS-2018-NewImgs'

            # Save preprocess data into output_dir
            output_dir = '/SSD/p76111262/CIC-IDS2018-New'

            return root_dir, output_dir
        elif database == 'CIC-IDS2018-v2-DoS':
            # folder that contains class labels
            root_dir = '/SSD/p76111262/CIC-IDS-2018-v2/DoS'

            # Save preprocess data into output_dir
            output_dir = '/SSD/p76111262/CIC-IDS2018-v2/DoS'

            return root_dir, output_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return '/SSD/ne6101157/C3D/c3d-pretrained.pth'