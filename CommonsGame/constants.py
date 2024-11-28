from enum import Enum

bigMap = [
    list('                                      '),
    list('             @      @@@@@       @     '),
    list('         @   @@         @@@    @  @   '),
    list('      @ @@@  @@@    @    @ @@ @@@@    '),
    list('  @  @@@ @    @  @ @@@  @  @   @ @    '),
    list(' @@@  @ @    @  @@@ @  @@@        @   '),
    list('  @ @  @@@  @@@  @ @    @ @@   @@ @@  '),
    list('   @ @  @@@    @ @  @@@    @@@  @     '),
    list('    @@@  @      @@@  @    @@@@        '),
    list('     @       @  @ @@@    @  @         '),
    list(' @  @@@  @  @  @@@ @    @@@@          '),
    list('     @ @   @@@  @ @      @ @@   @     '),
    list('      @@@   @ @  @@@      @@   @@@    '),
    list('  @    @     @@@  @             @     '),
    list('              @                       '),
    list('                                      '),
    list('              @                       '),
    list('              @                       '),
    list('              @                       '),
    list('              @                       '),
    list('              @                       '),
]

smallMap = [
    list('                     '),
    list('           @         '),
    list('          @@@        '),
    list('         @@@         '),
    list('          @          '),
    list('                     '),
    list('                     '),
    list('                     ')]

respawnProbs = [0.01, 0.05, 0.1]


class Actions(Enum):
    STEP_FORWARD = 0
    STEP_BACKWARD = 1
    STEP_LEFT = 2
    STEP_RIGHT = 3
    ROTATE_RIGHT = 4
    ROTATE_LEFT = 5
    STAND_STILL = 6
    TAG = 7

class Orientations(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3
    
class BeamDefs:
    WIDTH = 2
    HEIGHT = 10