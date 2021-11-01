import json
import numpy as np
"""
author:lyl
convert darknet result.json to TT100K json
"""
coco = dict()
coco['imgs'] = {}
with open("/home/dell/PycharmProjects/traffic_test/yolov4_result.json") as f:
    json_data=json.load(f)
    for json_data_ in json_data:
        image_id=json_data_['filename'].split("/")[-1].split(".jpg")[0]
        objects_=json_data_['objects']
        object_item = []
        if objects_==[]:
            # print(image_id)
            object_item = []
        else:
            for b in objects_:
                object_name=b['name']
                score_=b['confidence']
                relative_coordinates=b['relative_coordinates']
                coordinates_width=relative_coordinates['width']*2048
                coordinates_height = relative_coordinates['height'] * 2048
                coordinates_center_x=relative_coordinates['center_x'] * 2048
                coordinates_center_y= relative_coordinates['center_y'] * 2048
                #[x y w h]--->[xmin ymin xmax ymax]
                xmin = coordinates_center_x - coordinates_width / 2
                ymin= coordinates_center_y - coordinates_height  / 2
                xmax = coordinates_center_x + coordinates_width/ 2
                ymax = coordinates_center_y + coordinates_height  / 2
                bbox=dict()
                bbox['xmin']=xmin
                bbox['ymin'] = ymin
                bbox['ymax'] = ymax
                bbox['xmax'] = xmax
                cate_item=dict()
                cate_item['category']=object_name
                cate_item['score']=score_
                cate_item['bbox']=bbox
                object_item.append(cate_item)
        obv={"objects": object_item}
        dv={image_id:obv}
        coco['imgs'].update(dv)
with open("/home/dell/PycharmProjects/traffic_test/yolov4_record.json","w") as f:
     json.dump(coco,f)
     print("加载入文件完成...")
