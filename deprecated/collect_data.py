def observe_by_line(rob):
    visible = rob.getCachedObserve('getLineOfSights')
    if visible is not None:
        result = [visible,
              visible['x'],
              visible['y'],
              visible['z']]
    return result


def runSkill(rob, b):
    status = 'running'
    while status == 'running':
        rob.updateAllObservations()
        status, actions = b()
        for act in actions:
            rob.sendCommand(act)
        time.sleep(0.2)

def collect_data(rob):
    # up is negative, down positive
    pitch_t = [15, -5, 5]
    # right is positive, left is negative
    yaw_t = [-30, 0, 30]
    rob.updateAllObservations()
    pos = rob.waitNotNoneObserve('getAgentPos')
    current_pitch = pos[PITCH]
    current_yaw = pos[YAW]
    data = []
    for p in pitch_t:
        for y in yaw_t:
            b = TurnTo(rob, current_pitch + p, current_yaw + y)
            runSkill(rob, b)
            # turned to desired direction, collect point
            point = observe_by_line(rob)
            # collect frame
            frame = rob.getCachedObserve('getImageFrame')
            data.append((point, frame))
            print(point)
            print(numpy.asarray(frame.modelViewMatrix))

    b = TurnTo(rob, current_pitch, current_yaw)
    runSkill(rob, b)
    point = observe_by_line(rob)
    data.append((point, frame))
    return data
