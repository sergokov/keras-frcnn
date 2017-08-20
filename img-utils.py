import os
from optparse import OptionParser


def rename_img(path):
    dir_files = os.listdir(path)
    for i, f in enumerate(dir_files):
        os.rename(os.path.join(path ,f), os.path.join(path ,"coke-bottle-%s.%s" % (i, f.split(".")[1])))


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-a", "--action", dest="action", help="Please specify action to do: rename, analyze")
    parser.add_option("-p", "--dir-path", dest="dir_path", help="Please specify dir images path.")

    (options, args) = parser.parse_args()
    if options.action is None:
        raise Exception('Action arg should be specify.')

    if options.dir_path is None:
        raise Exception('dir-path arg should be specify.')

    if options.action in 'rename':
        rename_img(options.dir_path)