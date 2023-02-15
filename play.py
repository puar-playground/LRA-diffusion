import os
import shutil
from PIL import Image


def modify_data(in_dir):

    f = open(in_dir, 'r')
    f_out = open('/Users/chenjian/Desktop/working_space_LRA/LRA-diff_cifar/rvl-cdip/labels/test_new.txt', 'w')
    class_name_dict = {0: 'letter', 1: 'form', 2: 'email', 3: 'handwritten', 4: 'advertisement', 5: 'scientific_report',
                  6: 'scientific_publication', 7: 'specification', 8: 'file_folder', 9: 'news_article', 10: 'budget',
                  11: 'invoice', 12: 'presentation', 13: 'questionnaire', 14: 'resume', 15: 'memo'}

    for k in class_name_dict.values():
        print('/Users/chenjian/Desktop/working_space_LRA/LRA-diff_cifar/rvl-cdip/images_test/' + k)
        if not os.path.exists('/Users/chenjian/Desktop/working_space_LRA/LRA-diff_cifar/rvl-cdip/images_test/' + k):
            os.mkdir('/Users/chenjian/Desktop/working_space_LRA/LRA-diff_cifar/rvl-cdip/images_test/' + k)

    class_cnt = {i: 0 for i in range(16)}

    for line in f:
        old_dir = line.split(' ')[0]
        l = int(line[:-1].split(' ')[1])
        class_name = class_name_dict[l]
        batch = int(class_cnt[l] / 500)
        new_dir = os.path.join('/Users/chenjian/Desktop/working_space_LRA/LRA-diff_cifar/rvl-cdip/images_test/',
                               class_name, f'{batch:02d}')
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)

        new_dir_file = os.path.join(new_dir, str(class_cnt[l]) + '.tif')
        try:
            image = Image.open('/Users/chenjian/Desktop/working_space_LRA/LRA-diff_cifar/rvl-cdip/images/' + old_dir)
        except:
            print('/Users/chenjian/Desktop/working_space_LRA/LRA-diff_cifar/rvl-cdip/images/' + old_dir)
            print('discard')
            continue

        new_line = os.path.join('images_test/', class_name, f'{batch:02d}',
                                str(class_cnt[l]) + '.tif') + ' ' + line[:-1].split(' ')[1] + '\n'
        f_out.writelines(new_line)
        shutil.move('/Users/chenjian/Desktop/working_space_LRA/LRA-diff_cifar/rvl-cdip/images/' + old_dir, new_dir_file)
        class_cnt[l] += 1




if __name__ == "__main__":

    in_dir = '/Users/chenjian/Desktop/working_space_LRA/LRA-diff_cifar/rvl-cdip/labels/test.txt'
    modify_data(in_dir)