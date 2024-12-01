big_map = [
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

small_map = [
    list('                     '),
    list('           @         '),
    list('          @@@        '),
    list('         @@@         '),
    list('          @          '),
    list('                     '),
    list('                     '),
    list('                     ')]

class RespawnProbabilities:
    LOW = 0.01
    MEDIUM = 0.05
    HIGH = 0.1

class Colours:
    RED = [255, 0, 0]
    GREEN = [0, 255, 0]
    BLUE = [0, 0, 255]
    YELLOW = [255, 255, 0]
    BLACK = [0, 0, 0]
    GRAY = [52, 52, 52]
    WHITE = [180, 180, 180]

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
    GIFT = '$'
    SCOPE = '-'
    EMPTY = ' '
    AGENT = '%'

class BeamRange:
    WIDTH = 2
    HEIGHT = 10
    
class GiftRange:
    WIDTH = 2
    HEIGHT = 5
    
DEFAULT_COLOURS = [
    (Sprites.WALL, Colours.WHITE),
    (Sprites.APPLE, Colours.GREEN),
    (Sprites.BEAM, Colours.YELLOW),
    (Sprites.GIFT, Colours.BLUE),
    (Sprites.SCOPE, Colours.GRAY),
    (Sprites.EMPTY, Colours.BLACK)
]

COLOUR_TO_CHAR = {
    tuple(Colours.RED): Sprites.AGENT,
    tuple(Colours.GREEN): Sprites.APPLE,
    tuple(Colours.BLUE): Sprites.GIFT,
    tuple(Colours.YELLOW): Sprites.BEAM,
    tuple(Colours.GRAY): Sprites.SCOPE,
    tuple(Colours.BLACK): Sprites.EMPTY,
    tuple(Colours.WHITE): Sprites.WALL
}
    
TIMEOUT = 25
NUM_ORIENTATIONS = len([attr for attr in vars(Orientations) if not attr.startswith("__")])
NUM_ACTIONS = len([attr for attr in vars(Actions) if not attr.startswith("__")])
