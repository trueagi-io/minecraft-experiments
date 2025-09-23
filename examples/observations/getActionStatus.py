""""An agent can be controlled via sending commands.
    The commands of continuous movement which are being executed can be tracked
    using getActionStatus()."""
    
from tagilmo.utils.vereya_wrapper import MCConnector
import tagilmo.utils.mission_builder as mb
import time

miss = mb.MissionXML()
miss.setWorld(mb.flatworld("",
                           seed= '5',
              forceReset = "true"))
miss.setObservations(mb.Observations())

mc = MCConnector(miss)
mc.safeStart()

time.sleep(2)
print("Turn 0.5")
mc.turn("0.5")
time.sleep(0.1)
mc.observeProc()
print("actionStatus: ", mc.getActionStatus())
print("Move 1")
mc.sendCommand("move 1")
time.sleep(0.1)
mc.observeProc()
print("actionStatus: ", mc.getActionStatus())