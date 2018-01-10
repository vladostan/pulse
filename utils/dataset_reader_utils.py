import os


def load_dataset_from_ogg(name, Clip):
    clips = []
    for directory in sorted(os.listdir('{0}/'.format(name))):
        directory = '{0}/{1}'.format(name, directory)
        if os.path.isdir(directory) and os.path.basename(directory)[0:3].isdigit():
            print('Parsing ' + directory)
            category = []
            for clip in sorted(os.listdir(directory)):
                if clip[-3:] == 'ogg':
                    print ('{0}/{1}'.format(directory, clip))
                    category.append(Clip('{0}/{1}'.format(directory, clip), 'ogg'))
            clips.append(category)
    print('All {0} recordings loaded.'.format(name))
    return clips
