import torch
import math
from .builder import IOU_CALCULATORS


@IOU_CALCULATORS.register_module()
class BboxOverlaps2D(object):
    """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        """Calculate IoU between 2D bboxes.

        Args:
            bboxes1 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, or shape (m, 5) in <x1, y1, x2, y2, score> format.
            bboxes2 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, shape (m, 5) in <x1, y1, x2, y2, score> format, or be
                empty. If ``is_aligned `` is ``True``, then m and n must be
                equal.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground), or "giou" (generalized intersection over
                union).
            is_aligned (bool, optional): If True, then m and n must be equal.
                Default False.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        assert bboxes1.size(-1) in [0, 4, 5]
        assert bboxes2.size(-1) in [0, 4, 5]
        if bboxes2.size(-1) == 5:
            bboxes2 = bboxes2[..., :4]
        if bboxes1.size(-1) == 5:
            bboxes1 = bboxes1[..., :4]
        return bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + '()'
        return repr_str


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned `` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned `` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )

    Example:
        >>> empty = torch.empty(0, 4)
        >>> nonempty = torch.FloatTensor([[0, 0, 10, 9]])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'iof', 'giou', 'diou', 'ciou', 'dotd','iou_dotd'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes's last dimenstion is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])
    
    
    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]
        wh = (rb - lt).clamp(min=0)  # [B, rows, 2]
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou','iou_dotd']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
        if mode == 'diou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
            b1_x1, b1_y1 = bboxes1[..., 0], bboxes1[..., 1]
            b1_x2, b1_y2 = bboxes1[..., 2], bboxes1[..., 3]
            b2_x1, b2_y1 = bboxes2[..., 0], bboxes2[..., 1]
            b2_x2, b2_y2 = bboxes2[..., 2], bboxes2[..., 3]
        if mode == 'ciou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
            b1_x1, b1_y1 = bboxes1[..., 0], bboxes1[..., 1]
            b1_x2, b1_y2 = bboxes1[..., 2], bboxes1[..., 3]
            b2_x1, b2_y1 = bboxes2[..., 0], bboxes2[..., 1]
            b2_x2, b2_y2 = bboxes2[..., 2], bboxes2[..., 3]
        if mode == 'dotd':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
            b1_x1, b1_y1 = bboxes1[..., 0], bboxes1[..., 1]
            b1_x2, b1_y2 = bboxes1[..., 2], bboxes1[..., 3]
            b2_x1, b2_y1 = bboxes2[..., 0], bboxes2[..., 1]
            b2_x2, b2_y2 = bboxes2[..., 2], bboxes2[..., 3]
        if mode == 'iou_dotd':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
            b1_x1, b1_y1 = bboxes1[..., 0], bboxes1[..., 1]
            b1_x2, b1_y2 = bboxes1[..., 2], bboxes1[..., 3]
            b2_x1, b2_y1 = bboxes2[..., 0], bboxes2[..., 1]
            b2_x2, b2_y2 = bboxes2[..., 2], bboxes2[..., 3]
            

    else:
        lt = torch.max(bboxes1[..., :, None, :2],
                       bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:],
                       bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = (rb - lt).clamp(min=0)  # [B, rows, cols, 2]
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou','iou_dotd']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])
        if mode == 'diou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])
            b1_x1, b1_y1 = bboxes1[..., :, None, 0], bboxes1[..., :, None, 1]
            b1_x2, b1_y2 = bboxes1[..., :, None, 2], bboxes1[..., :, None, 3]
            b2_x1, b2_y1 = bboxes2[..., None, :, 0], bboxes2[..., None, :, 1]
            b2_x2, b2_y2 = bboxes2[..., None, :, 2], bboxes2[..., None, :, 3]
        if mode == 'ciou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])
            b1_x1, b1_y1 = bboxes1[..., :, None, 0], bboxes1[..., :, None, 1]
            b1_x2, b1_y2 = bboxes1[..., :, None, 2], bboxes1[..., :, None, 3]
            b2_x1, b2_y1 = bboxes2[..., None, :, 0], bboxes2[..., None, :, 1]
            b2_x2, b2_y2 = bboxes2[..., None, :, 2], bboxes2[..., None, :, 3]
        if mode == 'dotd':
            b1_x1, b1_y1 = bboxes1[..., :, None, 0], bboxes1[..., :, None, 1]
            b1_x2, b1_y2 = bboxes1[..., :, None, 2], bboxes1[..., :, None, 3]
            b2_x1, b2_y1 = bboxes2[..., None, :, 0], bboxes2[..., None, :, 1]
            b2_x2, b2_y2 = bboxes2[..., None, :, 2], bboxes2[..., None, :, 3]
           # union = area1[..., None] + area2[..., None, :] - overlap
        if mode == 'iou_dotd':
            b1_x1, b1_y1 = bboxes1[..., :, None, 0], bboxes1[..., :, None, 1]
            b1_x2, b1_y2 = bboxes1[..., :, None, 2], bboxes1[..., :, None, 3]
            b2_x1, b2_y1 = bboxes2[..., None, :, 0], bboxes2[..., None, :, 1]
            b2_x2, b2_y2 = bboxes2[..., None, :, 2], bboxes2[..., None, :, 3]

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)

    ious = overlap / union

    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    
    if mode == 'dotd' :
        left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2))**2 / 4  #== ((b2_x1 + b2_x2)/2 - (b1_x1 + b1_x2)/2)**2
        right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2))**2 / 4 
        rho2 = left + right
        i=0
        total=0
        for bbox in bboxes2:
            i=i+1
            total = total + (bbox[..., 2] - bbox[..., 0]) * (bbox[..., 3] - bbox[..., 1])
            st = (bbox[..., 2] - bbox[..., 0]) * (bbox[..., 3] - bbox[..., 1])
        total = total / i
        up = rho2 / total
        last = torch.sqrt(up)
        dotd = torch.exp(-1 * last)
        return dotd
    
    if mode == 'iou_dotd' :
        small_gt_mask = (area2 < 1024)
        large_gt_mask = (area2 >= 1024)
        small_gt_bboxes = bboxes2[small_gt_mask]
        large_gt_bboxes = bboxes2[large_gt_mask]
        bbox2_size = torch.zeros_like(area2)
        bbox2_size[small_gt_mask] = 0
        bbox2_size[large_gt_mask] = 1
        '''
        ioufull = torch.cat([ious.unsqueeze(-1),bbox2_size..unsqueeze(-1)],dim = 1)
        iou_dotd = torch.zeros_like(ioufull)
        iou_dotd[..., 0][small_gt_mask] = ious_small
        iou_dotd[..., 0][large_gt_mask] = dotd_large
        iou_dotd[..., 1] = bbox2_size
        '''
        left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2))**2 / 4  #== ((b2_x1 + b2_x2)/2 - (b1_x1 + b1_x2)/2)**2
        right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2))**2 / 4 
        rho2 = left + right
        i=0
        total=0
        for bbox in bboxes2:
            i=i+1
            total = total + (bbox[..., 2] - bbox[..., 0]) * (bbox[..., 3] - bbox[..., 1])
            st = (bbox[..., 2] - bbox[..., 0]) * (bbox[..., 3] - bbox[..., 1])
        total = total / i
        up = rho2 / total
        last = torch.sqrt(up)
        dotd = torch.exp(-1 * last)
        iou = ious
        iou_dotd = torch.zeros_like(ious)
        a= area2.size()
        b= iou.size()
        c= dotd.size()
        '''
        print("the area2 size is")
        print(a)
        print("the dotd size is")
        print(b)
        print("the iou size is")
        print(c)
        print("the lt is")
        print(lt)
        print("the wh is")
        print(wh)
        print("the overlap is")
        print(overlap)
        '''
        if small_gt_mask is not None:
            '''
            small_mask = (area2 < 322).unsqueeze(-1)
            print("small mask is")
            print(small_mask.size())
            large_mask = ~small_mask
            iou_dotd = torch.where(small_mask, dotd, iou)
            '''
            #print("we have entered")
            small_gt_mask_int = small_gt_mask.type(torch.int)
            large_gt_mask_int = large_gt_mask.type(torch.int)
            '''
            print("the area2 is")
            print(area2)
            print("the small_gt_mask_int is")
            print(small_gt_mask_int)
            print("the large_gt_mask_int is")
            print(large_gt_mask_int)
            print("the iou is")
            print(iou)
            print("the dotd is")
            print(dotd)
            '''
            iou_dotd = small_gt_mask_int * dotd + large_gt_mask_int * iou
            #iou_dotd = torch.where(area2<322,dotd,iou)
            #print("the iou_dotd is")
            #print(iou_dotd)
        else:
            iou_dotd = iou
        #d=iou_dotd.size()
        #print("the iou_dotd size is")
        #print(d)
        return iou_dotd
    
    if mode in ['giou']:
        enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
        enclose_area = torch.max(enclose_area, eps)
        gious = ious - (enclose_area - union) / enclose_area
        return gious
    if mode in ['ciou']:
        left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2))** 2 / 4
        right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2))** 2 / 4
        rho2 = left + right
        enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
        enclose_c = enclose_wh[..., 0]** 2 + enclose_wh[..., 1] ** 2
        enclose_c = torch.max(enclose_c, eps)

        factor = 4 / 3.1415926**2
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        v = factor * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
        cious = ious - (rho2 / enclose_c+v**2 / (1- ious + v))
        return cious
    if mode in ['diou']:
        left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2))**2 / 4
        right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2))**2 / 4
        rho2 = left + right
        enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
        enclose_c = enclose_wh[..., 0]**2 + enclose_wh[..., 1]**2
        enclose_c = torch.max(enclose_c, eps)
        dious = ious - rho2 / enclose_c
    return dious

