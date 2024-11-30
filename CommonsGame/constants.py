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
    list('             @                        '),
    list('              @                       '),
    list('              @                       '),
    list('              @                       '),
    list('              @                       '),
    list('              @                       '),
]

smallMap = [
    list('                     '),
    list('           =         '),
    list('          ===        '),
    list('         ===         '),
    list('          =          '),
    list('                     '),
    list('                     '),
    list('                     ')]

respawnProbs = [0.01, 0.05, 0.1]

class Actions:
    STEP_FORWARD = 0
    STEP_BACKWARD = 1
    STEP_LEFT = 2
    STEP_RIGHT = 3
    ROTATE_RIGHT = 4
    ROTATE_LEFT = 5
    STAND_STILL = 6
    TAG = 7
    GIFT = 8

class Orientations:
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3
    
class Sprites:
    WALL = '='
    APPLE = '@'
    AGENTS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    BEAM = '.'
    GIFT = '/'
    SCOPE = '-'
    EMPTY = ' '

class BeamDefs:
    WIDTH = 2
    HEIGHT = 10
    
    
NUM_ORIENTATIONS = len([attr for attr in vars(Orientations) if not attr.startswith("__")])
NUM_ACTIONS = len([attr for attr in vars(Actions) if not attr.startswith("__")])
