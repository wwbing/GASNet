import torch
from yolov6.utils.general import check_version 

torch_1_10_plus = check_version(torch.__version__, minimum='1.10.0')

# 候选框的生成
def generate_anchors(feats, fpn_strides, grid_cell_size=5.0, grid_cell_offset=0.5,  device='cpu', is_eval=False, mode='af'):
    '''Generate anchors from features.'''
    anchors = []
    anchor_points = []
    stride_tensor = []
    num_anchors_list = []
    assert feats is not None
    
    if is_eval:
        for i, stride in enumerate(fpn_strides):
            _, _, h, w = feats[i].shape
            shift_x = torch.arange(end=w, device=device) + grid_cell_offset
            shift_y = torch.arange(end=h, device=device) + grid_cell_offset
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij') if torch_1_10_plus else torch.meshgrid(shift_y, shift_x)
            anchor_point = torch.stack(
                    [shift_x, shift_y], axis=-1).to(torch.float)
            if mode == 'af': # anchor-free
                anchor_points.append(anchor_point.reshape([-1, 2]))
                stride_tensor.append(
                torch.full(
                    (h * w, 1), stride, dtype=torch.float, device=device))
            elif mode == 'ab': # anchor-based
                anchor_points.append(anchor_point.reshape([-1, 2]).repeat(3,1))
                stride_tensor.append(
                    torch.full(
                        (h * w, 1), stride, dtype=torch.float, device=device).repeat(3,1))
        anchor_points = torch.cat(anchor_points)
        stride_tensor = torch.cat(stride_tensor)
        return anchor_points, stride_tensor
    else:
        for i, stride in enumerate(fpn_strides):
            _, _, h, w = feats[i].shape
            cell_half_size = grid_cell_size * stride * 0.5
            
            # 特征图位置偏移
            shift_x = (torch.arange(end=w, device=device) + grid_cell_offset) * stride
            shift_y = (torch.arange(end=h, device=device) + grid_cell_offset) * stride
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij') if torch_1_10_plus else torch.meshgrid(shift_y, shift_x)
            
            # 生成anchor [h, w, 4]
            anchor = torch.stack(
                [
                    shift_x - cell_half_size, shift_y - cell_half_size,
                    shift_x + cell_half_size, shift_y + cell_half_size
                ],
                axis=-1).clone().to(feats[0].dtype)
            
            # 生成anchor_point [h, w, 2]
            anchor_point = torch.stack(
                [shift_x, shift_y], axis=-1).clone().to(feats[0].dtype)

            # 如果是anchor-free模式，anchor个数为[h*w, 4], anchor_point个数为[h*w, 2]
            # 如果是anchor-based模式，anchor个数为[h*w*3, 4], anchor_point个数为[h*w*3, 2]  变成三倍
            if mode == 'af': # anchor-free
                anchors.append(anchor.reshape([-1, 4]))
                anchor_points.append(anchor_point.reshape([-1, 2]))
            elif mode == 'ab': # anchor-based
                anchors.append(anchor.reshape([-1, 4]).repeat(3,1))
                anchor_points.append(anchor_point.reshape([-1, 2]).repeat(3,1))
            
            # 记录anchor个数
            num_anchors_list.append(len(anchors[-1]))

            # 记录stride [h*w, 1]  最后是[8400, 1] 其中6400个8，1600个16，400个32
            stride_tensor.append(
                torch.full(
                    [num_anchors_list[-1], 1], stride, dtype=feats[0].dtype))
        
        anchors = torch.cat(anchors)
        anchor_points = torch.cat(anchor_points).to(device)
        stride_tensor = torch.cat(stride_tensor).to(device)
        return anchors, anchor_points, num_anchors_list, stride_tensor
