import sys
import tagilmo.utils.mission_builder as mb
from tagilmo.utils.vereya_wrapper import MCConnector, RobustObserverWithCallbacks
from behaviour.behaviours import *
import logging
from mcdemoaux.vision.neural import NeuralWrapper, get_image
from mcdemoaux.vision.vis import Visualizer
from tagilmo.VereyaPython import setupLogger
import logging
import time

SCALE = 2
RESIZE = 1
HEIGHT = 240 * SCALE
WIDTH = 320 * SCALE


def runSkill(rob):
    skill = MountainScan(rob)
    while True:
        acts = []
        rob.updateAllObservations()
        ray = rob.cached['getLineOfSights']
        if skill.precond() and not skill.finished():
            acts = skill.act()
        if skill.finished():
            return
        logging.info(acts)
        for act in acts:
            rob.sendCommand(act)
        time.sleep(0.2)


def visualize(img, segm):
    if img is not None:
       visualizer('image', img.pixels)
    if segm is not None:
       visualizer('segm', segm.pixels)


def show_heatmaps(obs):
    segm_data = obs.getCachedObserve('getNeuralSegmentation')
    if not(segm_data is None):
        heatmaps, img = segm_data
        visualizer('coal_ore', (heatmaps[0, 3].cpu().detach().numpy() * 255).astype(numpy.uint8))

if __name__ == '__main__':
    mc = MCConnector.connect(name='Cristina', video=True)
    obs = RobustObserverWithCallbacks(mc)
    setupLogger()
    visualizer = Visualizer() 
    visualizer.start()
    show_img = lambda: visualize(None, obs.getCachedObserve('getImageFrame'))
    show_segm = lambda: visualize(obs.getCachedObserve('getSegmentationFrame'), None)
    neural_callback = NeuralWrapper(obs)
                                # cb_name, on_change event, callback
    obs.addCallback('getNeuralSegmentation', 'getImageFrame', neural_callback)

    obs.addCallback(None, 'getImageFrame', show_img)
    obs.addCallback(None, 'getSegmentationFrame', show_segm) 
    # attach visualization callback to getNeuralSegmentation
    obs.addCallback(None, 'getNeuralSegmentation', lambda: show_heatmaps(obs))
    runSkill(obs)
    visualizer.stop()
    sys.exit(0)
