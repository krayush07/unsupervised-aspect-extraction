import cPickle as pickle
from global_module.settings_module import set_dir

def generate_indexed_labels():
    label_hash = {}
    # input_file = open(set_dir.Directory('TR').label_filename).readlines()
    # curr_count = 0
    # for each_label in input_file:
    #     curr_label = each_label.strip()
    #     if not label_hash.has_key(curr_label):
    #         label_hash[curr_label] = curr_count
    #         curr_count += 1

    label_hash["joy"] = 0
    label_hash["sadness"] = 1
    label_hash["disgust"] = 2
    label_hash["anger"] = 3
    label_hash["fear"] = 4
    label_hash["surprise"] = 5
    label_hash["neutral"] = 6

    label_map_file = open(set_dir.Directory('TR').label_map_dict, 'wb')
    pickle.dump(label_hash, label_map_file, protocol=pickle.HIGHEST_PROTOCOL)

    print 'Total classes %d' %(len(label_hash))

def util():
    generate_indexed_labels()

def main():
    generate_indexed_labels()


if __name__ == '__main__':
    main()

