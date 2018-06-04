from global_module.settings_module import Directory

def extract(input_filename):
    output_file = open(input_filename + '_reviews.txt', 'w')
    lines = open(input_filename, 'r').readlines()
    for each_line in lines:
        if each_line.startswith('review/text:'):
            review = each_line[each_line.find('review/text:') + len('review/text:'):]
            if len(review) >= 20 and len(review) <= 275:
                output_file.write(review.strip() + '\n')

    output_file.close()


base_dir = Directory('TR').data_path
extract(base_dir + '/beer_advocate/beeradvocate.txt')
