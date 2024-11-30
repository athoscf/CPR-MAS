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
    list('              =                       '),
    list('             =                        '),
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

class RespawnProbabilities:
    LOW = 0.01
    MEDIUM = 0.05
    HIGH = 0.1

class Colors:
    RED = (999, 0, 0)
    GREEN = (0, 999, 0)
    BLUE = (0, 0, 999)
    YELLOW = (999, 999, 0)
    BLACK = (0, 0, 0)
    GRAY = (200, 200, 200)
    WHITE = (705, 705, 705)

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

class BeamRange:
    WIDTH = 2
    HEIGHT = 10
    
class GiftRange:
    WIDTH = 2
    HEIGHT = 5
    
NUM_ORIENTATIONS = len([attr for attr in vars(Orientations) if not attr.startswith("__")])
NUM_ACTIONS = len([attr for attr in vars(Actions) if not attr.startswith("__")])
