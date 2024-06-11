class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'CIC-IDS2018-v1-DoS':
            root_dir = '/SSD/p76111262/CIC-IDS-2018-v1/DoS'
            output_dir = '/SSD/p76111262/CIC-IDS2018-v1/DoS'
            
            return root_dir, output_dir

        elif database == 'CIC-IDS2018':
            root_dir = '/SSD/p76111262/CIC-IDS-2018-all'
            output_dir = '/SSD/p76111262/CIC-IDS2018-all'
            
            return root_dir, output_dir

        elif database == 'CIC-IDS2018-v2-DoS':
            root_dir = '/SSD/p76111262/CIC-IDS-2018-v2/DoS'
            output_dir = '/SSD/p76111262/CIC-IDS2018-v2/DoS'
            
            return root_dir, output_dir

        elif database == 'CIC-IDS2018-v3-DoS':
            root_dir = '/SSD/p76111262/CIC-IDS-2018-v3/DoS'
            output_dir = '/SSD/p76111262/CIC-IDS2018-v3/DoS'
            
            return root_dir, output_dir

        elif database == 'CIC-IDS2018-v3-DDoS':
            root_dir = '/SSD/p76111262/CIC-IDS-2018-v3/DDoS'
            output_dir = '/SSD/p76111262/CIC-IDS2018-v3/DDoS'
            
            return root_dir, output_dir

        elif database == 'CIC-IDS2018-v3-Auth':
            root_dir = '/SSD/p76111262/CIC-IDS-2018-v3/Auth'
            output_dir = '/SSD/p76111262/CIC-IDS2018-v3/Auth'
            
            return root_dir, output_dir

        elif database == 'CIC-IDS2018-v3-Web':
            root_dir = '/SSD/p76111262/CIC-IDS-2018-v3/Web'
            output_dir = '/SSD/p76111262/CIC-IDS2018-v3/Web'
            
            return root_dir, output_dir

        elif database == 'CIC-IDS2018-v3-Other':
            root_dir = '/SSD/p76111262/CIC-IDS-2018-v3/Other'
            output_dir = '/SSD/p76111262/CIC-IDS2018-v3/Other'
            
            return root_dir, output_dir

        elif database == 'CIC-IDS2018-v3-Intrusion':
            root_dir = '/SSD/p76111262/CIC-IDS-2018-v3/Intrusion'
            output_dir = '/SSD/p76111262/CIC-IDS2018-v3/Intrusion'
            
            return root_dir, output_dir

        elif database == 'CIC-IDS2018-ZSL-DDoS':
            output_dir = '/SSD/p76111262/CIC-IDS2018-ZSL/DDoS'

            return None, output_dir

        elif database == 'CIC-IDS2018-ZSL-DoS':
            output_dir = '/SSD/p76111262/CIC-IDS2018-ZSL/DoS'

            return None, output_dir

        elif database == 'CIC-IDS2018-ZSL-Auth':
            output_dir = '/SSD/p76111262/CIC-IDS2018-ZSL/Auth'

            return None, output_dir

        elif database == 'CIC-IDS2018-ZSL-Web':
            output_dir = '/SSD/p76111262/CIC-IDS2018-ZSL/Web'

            return None, output_dir

        elif database == 'CIC-IDS2018-ZSL-Other':
            output_dir = '/SSD/p76111262/CIC-IDS2018-ZSL/Other'
            
            return None, output_dir

        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError