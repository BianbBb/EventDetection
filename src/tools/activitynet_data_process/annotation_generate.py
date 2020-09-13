import json

anno_path = "../../../data/ActivityNet/video_info_19993.json"
new_anno_path = "../../../data/ActivityNet/video_info_new.json"


def filter_json(raw_anno):
    rst_dict = {}
    for k, v in raw_anno.items():
        duration = v['duration']
        start = duration * 0.4
        end = duration * 0.6
        for label in v['annotations']:
            if label['segment'][0] < start or label['segment'][1] > end:
                v['annotations'].remove(label)
    for k, v in raw_anno.items():
        if len(v['annotations']) > 0:
            rst_dict[k] = raw_anno[k]
    return rst_dict


if __name__ == '__main__':
    json_file = open(anno_path)
    raw_anno = json.load(json_file)
    json_file.close()
    rst = filter_json(raw_anno)

    rst_file = open(new_anno_path, 'w')
    json.dump(rst, rst_file)