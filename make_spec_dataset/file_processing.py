import os
def get_segments(path):
    
    """
    reads a text file containg time segments when both detectors were opperational 
    """
    with open(path,'r') as f:
        segment_list = f.read().split('\n')
    done_list = []
    for segment in segment_list:
        if segment == '':
            continue
        t_i = int(segment.split(' ')[0])
        t_f = int(segment.split(' ')[1])
        done_list.append((t_i,t_f))
    return done_list

def segments_files(detector, segment_list=None, data_path=None):
    
    """
        GWOSC file are 4096 second long since some segemtns are longer in duration.
        This function returns a list of files that span the segment of interest.
        
    """
    if data_path is None: 
        with open(f'O1_{detector}1.txt','r') as f:
            my_list = f.read().split('\n')
    else:
        with open(os.path.join(data_path, f'O1_{detector}1.txt'),'r') as f:
            my_list = f.read().split('\n')
    s3_list = []

    for item in my_list:
        s3_list .append(f'{detector}1_' + item.split('-')[-2] + '.hdf5')

    detector_segment_files = []
    for segment in segment_list:
        t_i = segment[0]
        t_f = segment[1]
        files = []
        for item in s3_list:
            if int(item[3:-5]) <= t_i and int(item[3:-5])+4096 >= t_i:
                files.append(item)
            if int(item[3:-5]) >= t_i and int(item[3:-5]) < t_f:
                files.append(item)
        detector_segment_files.append(files)
    return detector_segment_files
