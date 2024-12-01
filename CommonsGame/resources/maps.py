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
    num_agents = 9
    agent_chars = "ABCDEFGHI"
    map = [
        list('            @      @@@@@           @    '),
        list('        @   @@         @@@        @  @  '),
        list('     @ @@@  @@@    @    @  @ @@@ @@@@   '),
        list(' @  @@@ @    @  @ @@@  @  @@@ @   @ @   '),
        list('@@@  @ @    @  @@@ @  @@@  @ @       @  '),
        list(' @ @  @@@  @@@  @ @    @ @  @@@   @@ @@ '),
        list('  @@@  @    @ @  @@@    @@@  @     @@@  '),
        list('   @         @@@  @      @          @   '),
        list('  @@@    @    @               @    @@@  '),
        list('   @  @ @@@    @  @ @@@      @  @@@ @   '),
        list('  @  @@@ @    @  @@@ @      @@@  @ @    '),
        list(' @@@  @ @    @@@  @ @        @ @  @@@   '),
        list('  @ @  @@@    @ @  @@@        @@@  @    '),
        list('   @@@  @      @@@  @    @  @ @@@       '),
        list('    @       @  @ @@@    @  @@@ @        '),
        list('@  @@@  @  @  @@@ @    @@@  @ @     A   '),
        list('    @ @   @@@  @ @      @ @  @@@   B  C '),
        list('     @@@   @ @  @@@      @@@  @  I      '),
        list(' @    @     @@@  @        @       F E D '),
        list('             @                  H  G    '),
    ]