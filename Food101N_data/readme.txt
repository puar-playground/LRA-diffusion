If you use this data, please cite:

@inproceedings{lee2017cleannet,
  title={CleanNet: Transfer Learning for Scalable Image Classifier Training with Label Noise},
  author={Lee, Kuang-Huei and He, Xiaodong and Zhang, Lei and Yang, Linjun},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition ({CVPR})},
  year={2018}
}



Image thumbnails:
	images/ - All image files in jpeg. We resized the short edge is 320 and kept aspect ratio. 



Class Lists:
	meta/classes.txt - the list of 101 classes



Dataset Splits and Human Verifications:
	Definition of verification label:
	In the paper "CleanNet: Transfer Learning for Scalable Image Classifier Training with Label Noise", we proposed
	learning from human supervision to suppress label noise. Specifically, we manually mark whether an image-class
	pair is correct with a verification label.

	meta/imagelist.tsv - The list of all images [class/image_key]
	meta/verified_train.tsv - The image list, where the columns are [class/img_key, verificaiton_label].
							  verification_label=0 indicates the class is incorrect for the image.
							  verification_label=1 indicates the class is correct for the image.
							  The images also present in meta/imagelist.tsv.
	meta/verified_val.tsv - The image list, where the columns are [class/img_key, verificaiton_label].
							verification_label=0 indicates the class is incorrect for the image.
							verification_label=1 indicates the class is correct for the image.
							The images also present in meta/imagelist.tsv.



Held-out Class Lists:
	In the paper "CleanNet: Transfer Learning for Scalable Image Classifier Training with Label Noise", we explored
	using transfer learning to suppress label noise. Specifically, we only keep verification labels for part of the
	classes so that we only learn from human supervision for some of the classes and transfer the knowledge of label
	noise to other classes. For future research to follow the experiments in the paper, we include the held-out class
	lists in this dataset.

	meta/classes_heldout_10.txt - The list of 101 classes, where 10 classes are held-out (marked as 0, otherwise 1).
	meta/classes_heldout_30.txt - The list of 101 classes, where 30 classes are held-out (marked as 0, otherwise 1).
	meta/classes_heldout_50.txt - The list of 101 classes, where 50 classes are held-out (marked as 0, otherwise 1).
	meta/classes_heldout_70.txt - The list of 101 classes, where 70 classes are held-out (marked as 0, otherwise 1).

