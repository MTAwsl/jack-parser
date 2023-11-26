from enum import Enum
from io import TextIOBase 
from typing import List, Union, Dict

class OpCode(Enum):
    ADD = 0
    SUB = 1
    AND = 2
    OR = 3
    NOT = 4
    XOR = 5
    SHL = 6
    SHR = 7
    LOAD = 8 # LOAD ADDR UNUSED DST
    SAVE = 9 # SAVE ADDR SRC UNUSED
    PUSH = 10 # PUSH SRC UNUSED UNUSED
    POP = 11 # POP UNUSED UNUSED DST
    CALL = 12 # CALL SRC UNUSED UNUSED
    JE = 32
    JNE = 33
    JL = 34
    JLE = 35
    JG = 36
    JGE = 37
    JLS = 38
    JGS = 39

    # Virtual insns
    JMP = -1
    RET = -2
    JLES = -3
    JGES = -4
    MOV = -5
    LABEL = -6

class Register(Enum):
    R0 = 0
    R1 = 1
    R2 = 2
    R3 = 3
    R4 = 4
    R5 = 5
    RIP = 6
    RIO = 7

    def __int__(self):
        return self.value

class Reference:
    __slots__ = ('val',)
    def __init__(self):
        raise NotImplementedError("Base class Reference cannot be used.")
    def __str__(self):
        return str(self.val)
    def __int__(self):
        return int(self.val)

class ImmRef(Reference):
    def __init__(self, val):
        self.val = int(val) & 0xFF

class RegRef(Reference):
    def __init__(self, reg):
        self.val = Register[reg]

class Assembly:
    __slots__ = ("labels", "asm_code")
    def __init__(self, val):
        self.labels: Dict[str, int] = {}
        self.asm_code: List[int] = [] 
        if isinstance(val, TextIOBase):
            lines = val.readlines()
        elif isinstance(val, str):
            lines = val.splitlines()
        elif isinstance(val, list):
            lines = val
        else:
            raise TypeError(f"Type {type(val)} is not accepted.")
        
        _curr_pos = 0
        for line in lines:
            line = line.upper().strip()
            if line.startswith("LABEL"):
                args = line.split(" ")[1:]
                assert len(args) == 1
                assert args[0] not in self.labels
                assert args[0] not in OpCode._member_names_
                assert args[0] not in Register._member_names_
                assert args[0][0] not in [str(i) for i in range(10)]
                self.labels[args[0]] = _curr_pos
            elif line != "":
                if line.startswith("JLES") or line.startswith("JGES"):
                    _curr_pos += 8
                else:
                    _curr_pos += 4

        for line in lines:
            self.asm_code += self._parse_line(line.strip())

    def to_int_list(self):
        return self.asm_code.copy()

    def to_bytes(self):
        return bytes(self.asm_code)

    def _parse_line(self, insn: str) -> List[int]:
        if insn == "":
            return []

        _insn_splited = insn.split(' ')
        op_str = _insn_splited[0]
        args: List = _insn_splited[1:]
        op_code = OpCode[op_str.upper()]

        if op_code == OpCode.LABEL:
            return []

        for idx, refstr in enumerate(args):
            refstr = refstr.upper()
            if refstr.startswith("R"):
                args[idx] = RegRef(refstr)
            elif refstr in self.labels:
                args[idx] = ImmRef(self.labels[refstr])
            else:
                args[idx] = ImmRef(refstr)

        machine_code: List[int] = [0 for _ in range(4)]
        extended_code: List[int] = [] 

        match(op_code):
            case OpCode.JMP:
                # JMP Reg/Imm
                assert len(args) == 1
                machine_code[0] = OpCode.ADD.value
                if isinstance(args[0], ImmRef):
                    machine_code[0] |= 0x80

                # ADD SRC 0 RIP
                machine_code[1:4] = (int(args[0]), 0, 6)

            case OpCode.JLES:
                # COND Reg/Imm Reg/Imm DST
                assert len(args) == 3
                assert isinstance(args[2], ImmRef)

                machine_code[0] = op_code.JLS.value
                if isinstance(args[0], ImmRef):
                    machine_code[0] |= 0x80
                if isinstance(args[1], ImmRef):
                    machine_code[0] |= 0x40
                machine_code[1:4] = map(lambda x: int(x), args[0:3])

                extended_code = [op_code.JE.value, 0, 0, 0]
                if isinstance(args[0], ImmRef):
                    extended_code[0] |= 0x80
                if isinstance(args[1], ImmRef):
                    extended_code[0] |= 0x40
                extended_code[1:4] = map(lambda x: int(x), args[0:3])
                
            case OpCode.JGES:
                # COND Reg/Imm Reg/Imm DST
                assert len(args) == 3
                assert isinstance(args[2], ImmRef)

                machine_code[0] = op_code.JGS.value
                if isinstance(args[0], ImmRef):
                    machine_code[0] |= 0x80
                if isinstance(args[1], ImmRef):
                    machine_code[0] |= 0x40
                machine_code[1:4] = map(lambda x: int(x), args[0:3])

                extended_code = [op_code.JE.value, 0, 0, 0]
                if isinstance(args[0], ImmRef):
                    extended_code[0] |= 0x80
                if isinstance(args[1], ImmRef):
                    extended_code[0] |= 0x40
                extended_code[1:4] = map(lambda x: int(x), args[0:3])

            case OpCode.RET:
                # POP ADDR 0 RIP
                assert len(args) == 0
                machine_code = [OpCode.POP.value, 0, 0, 6]

            case OpCode.MOV:
                # MOV SRC DST
                assert len(args) == 2
                assert isinstance(args[1], RegRef)
                
                machine_code[0] = op_code.ADD.value
                if isinstance(args[0], ImmRef):
                    machine_code[0] |= 0x80
                machine_code[1:4] = (int(args[0]), 0, int(args[1]))
            
            # Machine insns
            case op_code if op_code.value in range(0,8):
                # Arithmetic Reg/Imm Reg/Imm DST
                assert len(args) == 3
                assert isinstance(args[2], RegRef)
                
                machine_code[0] = op_code.value
                if isinstance(args[0], ImmRef):
                    machine_code[0] |= 0x80
                if isinstance(args[1], ImmRef):
                    machine_code[0] |= 0x40
                machine_code[1:4] = map(lambda x: int(x), args[0:3])

            case op_code if op_code.value in range(32, 40):
                # COND Reg/Imm Reg/Imm DST
                assert len(args) == 3
                assert isinstance(args[2], ImmRef)

                machine_code[0] = op_code.value
                if isinstance(args[0], ImmRef):
                    machine_code[0] |= 0x80
                if isinstance(args[1], ImmRef):
                    machine_code[0] |= 0x40
                machine_code[1:4] = map(lambda x: int(x), args[0:3])

            case op_code if op_code in [OpCode.PUSH, OpCode.CALL]:
                # PUSH/CALL Reg/Imm None None
                assert len(args) == 1

                machine_code[0] = op_code.value
                if isinstance(args[0], ImmRef):
                    machine_code[0] |= 0x80
                machine_code[1:4] = (int(args[0]), 0, 0)

            case OpCode.POP:
                # POP None None Reg
                assert len(args) == 1
                assert isinstance(args[0], RegRef)

                machine_code[0] = op_code.value
                machine_code[1:4] = (0, 0, int(args[0]))

            case OpCode.LOAD:
                # LOAD ADDR None TARGET
                assert len(args) == 2
                assert isinstance(args[1], RegRef)

                machine_code[0] = op_code.value
                if isinstance(args[0], ImmRef):
                    machine_code[0] |= 0x80
                machine_code[1:4] = (int(args[0]), 0, int(args[1]))

            case OpCode.SAVE:
                # SAVE ADDR Reg/Imm None
                assert len(args) == 2

                machine_code[0] = op_code.value
                if isinstance(args[0], ImmRef):
                    machine_code[0] |= 0x80
                if isinstance(args[1], ImmRef):
                    machine_code[0] |= 0x40
                machine_code[1:4] = (int(args[0]), int(args[1]), 0)

        return machine_code + extended_code

if __name__ == "__main__":
    import sys
    match (len(sys.argv)):
        case 2:
            outfile = "a.out"
        case 3:
            outfile = sys.argv[2]
        case _:
            print("python legasm.py <file.asm> [outfile]")
            exit()
    with open(sys.argv[1], "r") as f:
        asm = Assembly(f)
    with open(outfile, "wb") as w:
        w.write(asm.to_bytes())
    
    code = asm.to_int_list()
    buf = ""
    for i in code:
        buf += f"{i} "
    print(buf.strip())
