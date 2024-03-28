class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'ucf101':
            # folder that contains class labels
            root_dir = 'c:\\Users\\brian\\Desktop\\AI\\C3D\\UCF-101'

            # Save preprocess data into output_dir
            output_dir = 'c:\\Users\\brian\\Desktop\\AI\\C3D\\ucf101'

            return root_dir, output_dir
        elif database == 'hmdb51':
            # folder that contains class labels
            root_dir = '/Path/to/hmdb-51'

            output_dir = '/path/to/VAR/hmdb51'
        elif database == 'pac4':
            # folder that contains class labels
            root_dir = '/SSD/ne6101157/pac4'

            # Save preprocess data into output_dir
            output_dir = '/SSD/ne6101157/pac4'

            return root_dir, output_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return '/SSD/ne6101157/C3D/c3d-pretrained.pth'