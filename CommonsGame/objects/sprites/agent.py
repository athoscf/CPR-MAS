import numpy as np
from pycolab.prefab_parts import sprites
from CommonsGame.constants import *

class Agent(sprites.MazeWalker):
            
    def __init__(self, corner, position, character, agent_chars):
        super(Agent, self).__init__(corner, position, character, impassable=['='] + list(agent_chars.replace(character, '')), confined_to_board=True)
        self.agent_chars = agent_chars
        self.orientation = np.random.choice(NUM_ORIENTATIONS)
        self.index= agent_chars.index(character)
        self.init_pos = position
        self.timeout = 0
        self.reward = 0
        self.handle_action = {
            Actions.STEP_FORWARD: self.step_forward,
            Actions.STEP_BACKWARD: self.step_backward,
            Actions.STEP_LEFT: self.step_left,
            Actions.STEP_RIGHT: self.step_right,
            Actions.ROTATE_RIGHT: self.rotate_right,
            Actions.ROTATE_LEFT: self.rotate_left,
            Actions.STAND_STILL: self.stand_still
        }

    def update(self, actions, board, layers, backdrop, things, the_plot):
        if actions is None: return
                
        if not self._visible:
            self.handle_timeout()
            return
        
        if self.tagged_by_agent(things):
            self.remove_from_board()
            return
        
        action = actions[self.index]
        
        if action in self.handle_action:
            self.handle_action[action](board, the_plot)

    def handle_timeout(self):
        if self.timeout == 0:
            self._teleport(self.init_pos)
            self._visible = True
        else:
            self.timeout -= 1

    def remove_from_board(self):
        self.timeout = TIMEOUT
        self._visible = False 

    def tagged_by_agent(self, things):
        return things[Sprites.BEAM].curtain[self.position[0], self.position[1]]

    def step_forward(self, board, the_plot):
        if self.orientation == Orientations.NORTH:
            self._north(board, the_plot)
        elif self.orientation == Orientations.EAST:
            self._east(board, the_plot)
        elif self.orientation == Orientations.SOUTH:
            self._south(board, the_plot)
        elif self.orientation == Orientations.WEST:
            self._west(board, the_plot)
            
    def step_backward(self, board, the_plot):
        if self.orientation == Orientations.NORTH:
            self._south(board, the_plot)
        elif self.orientation == Orientations.EAST:
            self._west(board, the_plot)
        elif self.orientation == Orientations.SOUTH:
            self._north(board, the_plot)
        elif self.orientation == Orientations.WEST:
            self._east(board, the_plot)
            
    def step_left(self, board, the_plot):
        if self.orientation == Orientations.NORTH:
            self._west(board, the_plot)
        elif self.orientation == Orientations.EAST:
            self._north(board, the_plot)
        elif self.orientation == Orientations.SOUTH:
            self._east(board, the_plot)
        elif self.orientation == Orientations.WEST:
            self._south(board, the_plot)

    def step_right(self, board, the_plot):
        if self.orientation == Orientations.NORTH:
            self._east(board, the_plot)
        elif self.orientation == Orientations.EAST:
            self._south(board, the_plot)
        elif self.orientation == Orientations.SOUTH:
            self._west(board, the_plot)
        elif self.orientation == Orientations.WEST:
            self._north(board, the_plot)

    def rotate_right(self, board, the_plot):
        self.orientation = (self.orientation + 1) % NUM_ORIENTATIONS

    def rotate_left(self, board, the_plot):
        self.orientation = (self.orientation - 1) % NUM_ORIENTATIONS

    def stand_still(self, board, the_plot):
        self._stay(board, the_plot)
    