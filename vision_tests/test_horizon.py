import cv2
from utils import common
from utils.common import *
import numpy
import time
from behaviour.behaviours import TurnTo, PITCH, YAW
from tagilmo.utils.vereya_wrapper import MCConnector, RobustObserverWithCallbacks
from tagilmo.utils.mathutils import degree2rad
from tagilmo.VereyaPython import setupLogger

# def observe_by_line(rob):
#     visible = rob.getCachedObserve('getLineOfSights')
#     if visible is not None:
#         result = [visible,
#               visible['x'],
#               visible['y'],
#               visible['z']]
#     return result
#
#
# def runSkill(rob, b):
#     status = 'running'
#     while status == 'running':
#         rob.updateAllObservations()
#         status, actions = b()
#         for act in actions:
#             rob.sendCommand(act)
#         time.sleep(0.2)

##This method is currently not in use##

# def collect_data(rob):
#     # up is negative, down positive
#     pitch_t = [15, -5, 5]
#     # right is positive, left is negative
#     yaw_t = [-30, 0, 30]
#     rob.updateAllObservations()
#     pos = rob.waitNotNoneObserve('getAgentPos')
#     current_pitch = pos[PITCH]
#     current_yaw = pos[YAW]
#     data = []
#     for p in pitch_t:
#         for y in yaw_t:
#             b = TurnTo(rob, current_pitch + p, current_yaw + y)
#             runSkill(rob, b)
#             # turned to desired direction, collect point
#             point = observe_by_line(rob)
#             # collect frame
#             frame = rob.getCachedObserve('getImageFrame')
#             data.append((point, frame))
#             print(point)
#             print(numpy.asarray(frame.modelViewMatrix))
#
#     b = TurnTo(rob, current_pitch, current_yaw)
#     runSkill(rob, b)
#     point = observe_by_line(rob)
#     data.append((point, frame))
#     return data

def to_opengl(vec):
    """
    x and z axis are flipped
    """
    res = vec.copy()
    res[0] *= -1
    res[2] *= -1
    return res


def vec2screen(pt, pitch, yaw, perspective, winWidth, winHeight):
    """
    project vector from camera coordinates to pixel coordinates

    pt: list[Int]
      3d point in camera reference frame with malmo coordinates
    pitch: float
      camera rotation around x axis(radians)
    yaw: float
      camera rotation around y axis(radians)
    perspective: numpy.array
      opengl perspective projection matrix 4x4
    """
    pt_c = numpy.asarray(pt) 
    # change axis from mincraft to right-hand side rule expected by the rotation matrix
    # z -> x, x -> y, y -> z
    pt_c = pt_c[[2, 0, 1]] 
    # apply camera rotation
    R = rotation_matrix(0, pitch, yaw)
    pt_c_r = R @ pt_c
    # to minecraft
    # z -> y, x -> z, y -> x 
    pt_c_r = pt_c_r[[1, 2, 0]]
    # apply perspective matrix
    # derivation https://www.songho.ca/opengl/gl_transform.html
    pt_c_gl = numpy.append(to_opengl(pt_c_r), [1])
    pt_clip = perspective @ pt_c_gl
    y = pt_clip[1] / pt_clip[-1]
    x = pt_clip[0] / pt_clip[-1]
    z = pt_clip[2] / pt_clip[-1]
    h_half = winHeight // 2
    w_half = winWidth // 2
    y_w = h_half * y + (-1 + h_half)
    x_w = w_half * x + (-1 + w_half)
    # don't need z right now
    # f and n are near and far plane of perspective frustum
    # they can be computed from perspective matrix - see derivation
    # z_w = (f - n) / 2 * z + (f + n) / 2

    point = [winWidth - round(x_w), winHeight - round(y_w)]
    return point

def show_horizon(rob, height=64, horizon_dist=200):
    """
    draw horizon line on images from minecraft

    Parameters
    ----------
    rob: RobustObserver
    height: int 
      ground level in minecraft coordinates,
      64 is good default value for most default worlds
    horizon_dist: int
      aprox. render distance in blocks
      this increase this parameter together with render distance
    """
    rob.updateAllObservations()
    time.sleep(0.15)
    frame = rob.getCachedObserve('getImageFrame')
    if frame is not None:
        image = cv2.UMat(frame.pixels)
    else:
        return
    # transpose since opengl uses column,row and numpy uses row,column matrix representation
    perspective_matrix = common.perspective_gl[85].T
    camera_coords = [frame.xPos, frame.yPos + 1.62, frame.zPos]
    pitch, yaw = frame.pitch, frame.yaw
    pitch, yaw = degree2rad(pitch), degree2rad(yaw)
    # point on horizon line
    pt4 = numpy.zeros(3, dtype=numpy.float32)
    # set z to a point at render distance
    # minecraft renders only a small distance by default
    pt4[2] = horizon_dist
    # set y to horizon level
    pt4[1] = height - camera_coords[1]
    pt4 = vec2screen(pt4, -pitch, 0, perspective_matrix, frame.iWidth, frame.iHeight)

    # 0th row is top of the image, so going in negative direction
    # corresponds to going up
    if pt4[1] < 0: # all points below the horizon
        print('below')
        pt4[1] = 0
    if frame.iHeight <= pt4[1]: # all points above the horizon
        print('above')
        pt4[1] = frame.iHeight - 1
    image = cv2.line(image, (0, pt4[1]), (frame.iWidth, pt4[1]), (0, 255, 0), 2)
    cv2.imshow('img', image)
    cv2.waitKey(300) 


def main():
    mc = MCConnector.connect(name='Cristina', video=True)
    rob = RobustObserverWithCallbacks(mc)
    setupLogger()
    while True:
        show_horizon(rob)

if __name__ == '__main__':
    main()
