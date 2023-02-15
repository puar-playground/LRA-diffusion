# Clothing1M Dataset

This dataset is used in the CVPR15 paper [*Learning from Massive Noisy Labeled Data for Image Classification*](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Xiao_Learning_From_Massive_2015_CVPR_paper.pdf).

It contains roughly one million clothing images crawled from the Internet. Most of them have noisy labels, which are extracted from their surrounding texts. A few of them have clean labels, which are manually annotated. Details are listed below.

## Dataset details

**Categories.** All the images are categorized into 14 classes, numbered from 0 to 13. The corresponding class names are listed in `category_names_eng.txt` and `category_names_chn.txt` (the Chinese version).

**Data format.** The noisy and clean labeled data are stored in `noisy_label_kv.txt` and  `clean_label_kv.txt`, respectively. Both files are formatted as

    image_file_1 label_1
    image_file_2 label_2
    ...

**Protocols.** The noisy labeled data are used only for training, while the clean labeled data are split into training, validation, and test subsets. The images selected for each subset are listed in

    -  noisy_train_key_list.txt
    -  clean_train_key_list.txt
    -  clean_val_key_list.txt
    -  clean_test_key_list.txt

All the labels can be retrieved from either the `noisy_label_kv.txt` or the `clean_label_kv.txt`. Notice that some images have both noisy and clean labels. The number of samples in each subset is shown below.

![Venn graph](venn.png "Number of samples in each subset")

## Terms of use

By downloading the Clothing1M dataset, you agree to the following terms:

1.  You will use the data only for non-commercial research and educational purposes.
2.  You will **NOT** distribute the dataset.
3.  The Chinese University of Hong Kong makes no representations or warranties regarding the data. All rights of the images reserved by Baidu Inc. or the original owners.
4.  You accept full responsibility for your use of the data and shall defend and indemnify The Chinese University of Hong Kong, including their employees, officers and agents, against any and all claims arising from your use of the data, including but not limited to your use of any copies of copyrighted images that you may create from the data.

## Citation

    @inproceedings{xiao2015learning,
      title={Learning from Massive Noisy Labeled Data for Image Classification},
      author={Xiao, Tong and Xia, Tian and Yang, Yi and Huang, Chang and Wang, Xiaogang},
      booktitle={CVPR},
      year={2015}
    }