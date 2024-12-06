class SmallMap:
    num_agents = 1
    agent_chars = "A"
    map = [
        list('  @    @    @    @    @ '),
        list(' @@@  @@@  @@@  @@@  @@@'),
        list('  @    @    @    @    @ '),
        list('                        '),
        list('    @    @    @    @    '),
        list('   @@@  @@@  @@@  @@@   '),
        list('    @    @    @    @   A'),
    ]

class OpenMap:
    num_agents = 4
    agent_chars = "ABCD"
    map = [
        list('  @  @@@ @    @  @@@ @      @@@  @ @    '),
        list(' @@@  @ @    @@@  @ @        @ @  @@@   '),
        list('  @ @  @@@    @ @  @@@        @@@  @    '),
        list('   @@@  @      @@@  @    @  @ @@@       '),
        list('    @       @  @ @@@    @  @@@ @        '),
        list('@  @@@  @  @  @@@ @    @@@  @ @     A   '),
        list('    @ @   @@@  @ @      @ @  @@@   B  C '),
        list('     @@@   @ @  @@@      @@@  @  D      '),
        list(' @    @     @@@  @        @             '),
        list('             @                          '),
    ]
    
class OpenMapV2:
    num_agents = 6
    agent_chars = "ABCDEF"
    map = [
        list('       @  @@@ @      @@@  @ @    '),
        list('      @@@  @ @@@                 '),
        list(' @@    @      @        @@@  @    '),
        list('           @                     '),
        list('     @  @ @@@    @ @@@ @         '),
        list(' @     @@@ @    @@@ @ @    A     '),
        list('        @           @      B  C  '),
        list('@@@ @ @  @@@       @@@  @  D     '),
        list(' @ @  @@@  @        @       E    '),
        list('      @      @    @@@        F   '),
    ]
    
class SingleEntranceMap:
    num_agents = 6
    agent_chars = "ABCDEF"
    map = [
        list('     @    @    @    @   =   =   @    @    @    @     '),
        list('    @@@  @@@  @@@  @@@= =   ===@@@  @@@  @@@  @@@    '),
        list('     @    @    @    @ = =       @    @    @    @     '),
        list('======================= =   ========================='),
        list('               B                D                    '),
        list('            A                      E                 '),
        list('                 C              F                    '),
        list('=========================   = ======================='),
        list('     @    @    @    @       = = @    @    @    @     '),
        list('    @@@  @@@  @@@  @@@===   = =@@@  @@@  @@@  @@@    '),
        list('     @    @    @    @   =   =   @    @    @    @     '),
    ]
    
class UnequalEntranceMap:
    num_agents = 6
    agent_chars = "ABCDEF"
    map = [
        list('   @   @   @   @        =   =   @    @    @    @     '),
        list('  @@@ @@@ @@@ @@@     = =   ===@@@  @@@  @@@  @@@    '),
        list('   @   @   @   @      = =       @    @    @    @     '),
        list('======================= =   ========================='),
        list('               B                D                    '),
        list('            A                      E                 '),
        list('                 C              F                    '),
        list('=========================   = ======================='),
        list('     @    @    @    @       = = @    @    @    @     '),
        list('    @@@  @@@  @@@  @@@===   = =@@@  @@@  @@@  @@@    '),
        list('     @    @    @    @   =   =   @    @    @    @     '),
    ]
    
class MultipleEntranceMap:
    num_agents = 6
    agent_chars = "ABCDEF"
    map = [
        list('     @    @    @    @   =   =   @    @    @    @     '),
        list('    @@@  @@@  @@@  @@@= =   ===@@@  @@@  @@@  @@@    '),
        list('     @    @    @    @ = =       @    @    @    @     '),
        list('=  ==== =============== =   ======  ===========  ===='),
        list('               B                D                    '),
        list('            A                      E                 '),
        list('                 C              F                    '),
        list('==== ======= ============   = ============= ====== =='),
        list('     @    @    @    @       = = @    @    @    @     '),
        list('    @@@  @@@  @@@  @@@===   = =@@@  @@@  @@@  @@@    '),
        list('     @    @    @    @   =   =   @    @    @    @     '),
    ]
    
class RegionMap:
    num_agents = 6
    agent_chars = "ABCDEF"
    map = [
        list('     @    @    @    @           @    @    @    @     '),
        list('    @@@  @@@  @@@  @@@         @@@  @@@  @@@  @@@    '),
        list('     @    @    @    @           @    @    @    @     '),
        list('                                                     '),
        list('       C                                  F          '),
        list('   A        D                      E             B   '),
        list('                                                     '),
        list('                                                     '),
        list('     @    @    @    @           @    @    @    @     '),
        list('    @@@  @@@  @@@  @@@         @@@  @@@  @@@  @@@    '),
        list('     @    @    @    @           @    @    @    @     '),
    ]