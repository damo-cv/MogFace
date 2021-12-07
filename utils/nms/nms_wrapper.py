from . import cython_nms

def nms(dets, thresh):
    if dets.shape[0] == 0:
        return []
    return cython_nms.nms(dets, thresh)
