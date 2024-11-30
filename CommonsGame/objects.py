import numpy as np
from pycolab.prefab_parts import sprites
from pycolab import things as pythings
from scipy.ndimage import convolve
from CommonsGame.constants import *





                


class ShotDrape(pythings.Drape):
    """Tagging ray Drap"""
    def __init__(self, curtain, character, agentChars, numPadPixels):
        super().__init__(curtain, character)
        self.agentChars = agentChars
        self.numPadPixels = numPadPixels
        self.h = curtain.shape[0] - (numPadPixels * 2 + 2)
        self.w = curtain.shape[1] - (numPadPixels * 2 + 2)
        self.scopeHeight = numPadPixels + 1

    def update(self, actions, board, layers, backdrop, things, the_plot):
        beamWidth = BeamRange.WIDTH
        beamHeight = BeamRange.HEIGHT
        np.logical_and(self.curtain, False, self.curtain)
        if actions is not None:
            for i, a in enumerate(actions):
                if a == Actions.TAG.value:
                    agent = things[self.agentChars[i]]
                    if agent.visible:
                        pos = agent.position
                        if agent.orientation == Orientations.NORTH.value:
                            if np.any(layers['='][pos[0] - beamHeight:pos[0],
                                      pos[1] - beamWidth:pos[1] + beamWidth + 1]):
                                collisionIdxs = np.argwhere(layers['='][pos[0] - beamHeight:pos[0],
                                                            pos[1] - beamWidth:pos[1] + beamWidth + 1])
                                beamHeight = beamHeight - (np.max(collisionIdxs) + 1)
                            self.curtain[pos[0] - beamHeight:pos[0],
                            pos[1] - beamWidth:pos[1] + beamWidth + 1] = True
                        elif agent.orientation == Orientations.EAST.value:
                            if np.any(layers['='][pos[0] - beamWidth:pos[0] + beamWidth + 1,
                            pos[1] + 1:pos[1] + beamHeight + 1]):
                                collisionIdxs = np.argwhere(layers['='][pos[0] - beamWidth:pos[0] + beamWidth + 1,
                            pos[1] + 1:pos[1] + beamHeight + 1])
                                beamHeight = np.min(collisionIdxs)
                            self.curtain[pos[0] - beamWidth:pos[0] + beamWidth + 1,
                            pos[1] + 1:pos[1] + beamHeight + 1] = True
                        elif agent.orientation == Orientations.SOUTH.value:
                            if np.any(layers['='][pos[0] + 1:pos[0] + beamHeight + 1,
                            pos[1] - beamWidth:pos[1] + beamWidth + 1]):
                                collisionIdxs = np.argwhere(layers['='][pos[0] + 1:pos[0] + beamHeight + 1,
                            pos[1] - beamWidth:pos[1] + beamWidth + 1])
                                beamHeight = np.min(collisionIdxs)
                            self.curtain[pos[0] + 1:pos[0] + beamHeight + 1,
                            pos[1] - beamWidth:pos[1] + beamWidth + 1] = True
                        elif agent.orientation == Orientations.WEST.value:
                            if np.any(layers['='][pos[0] - beamWidth:pos[0] + beamWidth + 1,
                                      pos[1] - beamHeight:pos[1]]):
                                collisionIdxs = np.argwhere(layers['='][pos[0] - beamWidth:pos[0] + beamWidth + 1,
                                                            pos[1] - beamHeight:pos[1]])
                                beamHeight = beamHeight - (np.max(collisionIdxs) + 1)
                            self.curtain[pos[0] - beamWidth:pos[0] + beamWidth + 1, pos[1] - beamHeight:pos[1]] = True
                        # self.curtain[:, :] = np.logical_and(self.curtain, np.logical_not(layers['=']))
        else:
            return


