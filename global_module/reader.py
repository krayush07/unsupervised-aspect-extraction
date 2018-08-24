import numpy as np

class DataReader:
    def get_index_string(self, utt, word_dict, params):
        index_string = ''
        for each_token in utt.split():
            if (params.all_lowercase):
                if (word_dict.has_key(each_token.lower())):
                    each_token = each_token.lower()
                elif (word_dict.has_key(each_token)):
                    each_token = each_token
                elif (word_dict.has_key(each_token.title())):
                    each_token = each_token.title()
                elif (word_dict.has_key(each_token.upper())):
                    each_token = each_token.upper()
                else:
                    each_token = each_token.lower()

            index_string += str(word_dict.get(each_token, word_dict.get("unk"))) + '\t'
        return len(index_string.strip().split()), index_string.strip()

    def pad_string(self, id_string, curr_len, max_seq_len):
        id_string = id_string.strip() + '\t'
        while curr_len < max_seq_len:
            id_string += '0\t'
            curr_len += 1
        return id_string.strip()

    def format_string(self, inp_string, curr_string_len, max_len):
        if curr_string_len > max_len:
            # print('Maximum SEQ LENGTH reached. Stripping extra sequence.\n')
            op_string = '\t'.join(inp_string.split('\t')[:max_len])
        else:
            op_string = self.pad_string(inp_string, curr_string_len, max_len)
        return op_string

    def generate_id_map(self, params, data_filename, index_arr, dict_obj):
        global curr_label
        data_file_arr = open(data_filename, 'r').readlines()

        pos_seq_arr = []
        neg_seq_arr = []
        pos_seq_len = []
        neg_seq_len = []

        for each_idx in index_arr:
            curr_line = data_file_arr[each_idx].strip().split("\t")

            pos_line = curr_line[0]
            curr_pos_seq_token_string, curr_pos_seq_len = self.get_seq_map(dict_obj, params, pos_line)
            pos_seq_arr.append(curr_pos_seq_token_string)
            pos_seq_len.append(curr_pos_seq_len)

            if len(curr_line) > 1:
                neg_line = curr_line[1]
                curr_neg_seq_token_string, curr_neg_seq_len = self.get_seq_map(dict_obj, params, neg_line)
                neg_seq_arr.append(curr_neg_seq_token_string)
                neg_seq_len.append(curr_neg_seq_len)

        return pos_seq_arr, pos_seq_len, neg_seq_arr, neg_seq_len

    def get_seq_map(self, dict_obj, params, input_line):
        curr_seq_token_len, curr_seq_token_id = self.get_index_string(input_line, dict_obj.word_dict, params)
        curr_seq_token_string = self.format_string(curr_seq_token_id, curr_seq_token_len, params.MAX_SEQ_LEN)
        return curr_seq_token_string, curr_seq_token_len

    def data_iterator(self, params, data_filename, index_arr, dict_obj):
        pos_seq_arr, pos_seq_len, neg_seq_arr, neg_seq_len = self.generate_id_map(params, data_filename, index_arr, dict_obj)

        batch_size = params.batch_size
        num_batches = len(index_arr) / params.batch_size

        for i in range(num_batches):
            curr_pos_seq_arr, curr_pos_seq_len = self.get_current_batch(batch_size, i, pos_seq_arr, pos_seq_len)

            if params.mode == 'TR' or params.mode == 'VA':
                curr_neg_seq_arr, curr_neg_seq_len = self.get_current_batch(batch_size, i, neg_seq_arr, neg_seq_len)
                yield (curr_pos_seq_arr, curr_pos_seq_len, curr_neg_seq_arr, curr_neg_seq_len)

            elif params.mode == 'TE':
                yield (curr_pos_seq_arr, curr_pos_seq_len)

    def get_current_batch(self, batch_size, i, inp_seq_arr, inp_seq_len):
        curr_pos_seq_arr = np.loadtxt(inp_seq_arr[i * batch_size: (i + 1) * batch_size], delimiter='\t', dtype=np.int32)
        if (batch_size == 1):
            curr_pos_seq_arr = np.expand_dims(curr_pos_seq_arr, axis=0)
        curr_pos_seq_len = np.array(inp_seq_len[i * batch_size: (i + 1) * batch_size], dtype=np.int32)
        return curr_pos_seq_arr, curr_pos_seq_len