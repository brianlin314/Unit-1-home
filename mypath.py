class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'CIC-IDS2018-v1-DoS':
            # folder that contains class labels
            root_dir = '/SSD/p76111262/CIC-IDS-2018-v1/DoS'

            # Save preprocess data into output_dir
            output_dir = '/SSD/p76111262/CIC-IDS2018-v1/DoS'

            return root_dir, output_dir
        elif database == 'CIC-IDS2018-v2-DoS':
            # folder that contains class labels
            root_dir = '/SSD/p76111262/CIC-IDS-2018-v2/DoS'

            # Save preprocess data into output_dir
            output_dir = '/SSD/p76111262/CIC-IDS2018-v2/DoS'

            return root_dir, output_dir
        elif database == 'CIC-IDS2018-v3-DoS':
            # folder that contains class labels
            root_dir = '/SSD/p76111262/CIC-IDS-2018-v3/DoS'

            # Save preprocess data into output_dir
            output_dir = '/SSD/p76111262/CIC-IDS2018-v3/DoS'

            return root_dir, output_dir
        elif database == 'CIC-IDS2018-v3-DDoS':
            # folder that contains class labels
            root_dir = '/SSD/p76111262/CIC-IDS-2018-v3/DDoS'

            # Save preprocess data into output_dir
            output_dir = '/SSD/p76111262/CIC-IDS2018-v3/DDoS'

            return root_dir, output_dir
        elif database == 'CIC-IDS2018-v3-Auth':
            # folder that contains class labels
            root_dir = '/SSD/p76111262/CIC-IDS-2018-v3/Auth'

            # Save preprocess data into output_dir
            output_dir = '/SSD/p76111262/CIC-IDS2018-v3/Auth'

            return root_dir, output_dir
        elif database == 'CIC-IDS2018-v3-Web':
            # folder that contains class labels
            root_dir = '/SSD/p76111262/CIC-IDS-2018-v3/Web'

            # Save preprocess data into output_dir
            output_dir = '/SSD/p76111262/CIC-IDS2018-v3/Web'

            return root_dir, output_dir
        elif database == 'CIC-IDS2018-v3-Other':
            # folder that contains class labels
            root_dir = '/SSD/p76111262/CIC-IDS-2018-v3/Other'

            # Save preprocess data into output_dir
            output_dir = '/SSD/p76111262/CIC-IDS2018-v3/Other'

            return root_dir, output_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError