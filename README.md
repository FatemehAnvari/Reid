# Reid
For Reid model:
1.python extract_feature.py --data_dir 'C:\\Users\\Mavara\\Desktop' --query 'D1_test\\frame_sequence_normal\\D1' --gallery 'D3_test\\frame_sequence_normal\\D3'
--data_dir: main data directory 
--query: path of query folder
--gallery: path of gallery folder
#for example:
#query_path = 'C:\\Users\\Mavara\\Desktop\\D1_test\\frame_sequence_normal\\D1'
#gallery_path = 'C:\\Users\\Mavara\\Desktop\\D3_test\\frame_sequence_normal\\D3'

2.python reid_camera.py --feature_file_dir path mat file(feature_file) --camQ name camera Query
--camG name camera Gallery 


