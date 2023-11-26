from typing import List, Dict, Union

# Keywords must be seperated by space or symbol
keywords = [
    "class", "constructor", "function", "method", 
    "field", "static", "var", "int", "char", "boolean",
    "void", "true", "false", "null", "this", "let", "do",
    "if", "else", "while", "return",
]

symbol = list("{}()[].,;+-*/&|<>=~")

# int constant: [0-9]+
# string constant: "[\x20-\x7E]+"
# identifier: [A-Za-z_][A-Za-z0-9_]+, must be seperated by space

class DFA:
    def __init__(self, tree):
        self.tree = tree
    
    def match_short(self, s: str):
        node = self.tree
        path = ""

        assert(s != "")
        
        ch = s[0]
        while ch in node.keys():
            if len(s) == 1:
                if "" in node[ch]:
                    return path+ch
                return False

            if "" in node.keys():
                return path

            node = node[ch]
            path += ch
            
            ch = s[1]
            s = s[1:]

    def match_long(self, s: str):
        node = self.tree
        path = ""

        assert(s != "")
        
        ch = s[0]
        while ch in node.keys():
            if len(s) == 1:
                if "" in node[ch]:
                    return path+ch
                return False

            node = node[ch]
            path += ch

            ch = s[1]
            s = s[1:]

        return path if "" in node else False

    def matched(self, s: str) -> bool:
        if self.match_long(s) == s:
            return True
        if self.match_short(s) == s:
            return True
        return False


class KeywordsPattern(DFA):

    @staticmethod
    def _factor_char(words: List[str]):
        factored_words = {}

        for w in words:
            if w == "":
                factored_words[""] = True  # Accepted
                continue
            if w[0] in factored_words.keys():
                factored_words[w[0]].append(w[1:])
                continue
            factored_words[w[0]] = [w[1:]]

        for factor, new_words in factored_words.items():
            if new_words != True:
                factored_words[factor] = KeywordsPattern._factor_char(new_words)

        return factored_words 

    def __init__(self, words: List[str]):
        super().__init__(KeywordsPattern._factor_char(words))

    def keywords(self, path="", node=None, keywords=[]):
        if node is None:
            node = self.tree

        for k,v in node.items():
            if k == "":
                keywords.append(path)
                continue
            keywords = self.keywords(path+k, v)

        return keywords

class SymbolPattern(DFA):
    def __init__(self, symbols: List[str]):
        all_accepted = {"": True}
        tree = {}
        for ch in symbols:
            tree[ch] = all_accepted
        super().__init__(tree)

class IdentifierPattern(DFA):
    # identifier: [A-Za-z_][A-Za-z0-9_]+, must be seperated by space
    def __init__(self):
        tree = {}
        accept_state: Dict[str, Union[dict, bool]] = {"": True}
        accept_state2: Dict[str, Union[dict, bool]] = {"": True}
        for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_":
            tree[ch] = accept_state
            accept_state[ch] = accept_state2 
            accept_state2[ch] = accept_state2 
        for ch in "0123456789":
            accept_state2[ch] = accept_state2

        super().__init__(tree)

class StrConstPattern(DFA):
    # string constant: "[\x20-\x7E]+"
    def __init__(self):

        accept_state: Dict[str, bool] = {"": True}
        str_state: Dict[str, dict] = {}
        tree: Dict[str, dict] = {"\"": str_state}
        escape_state: Dict[str, dict] = {"n": str_state, "b": str_state, "f": str_state, "t": str_state, "\n": str_state}

        for i in range(0x20, 0x7F):
            str_state[chr(i)] = str_state

        str_state['\\'] = escape_state
        str_state['\"'] = accept_state 
        
        super().__init__(tree)

    def match_short(self, s: str):
        return self.match_long(s)
    
    def match_long(self, s: str):
        node = self.tree
        path = ""
        raw_length = 0

        assert(s != "")
        
        ch = s[0]
        while ch in node.keys():
            if len(s) == 1:
                if "" in node[ch]:
                    return path+ch, raw_length+1
                return False

            node = node[ch]
            if ch != '\\':
                path += ch
                ch = s[1]
                s = s[1:]
                raw_length += 1
            else:
                escape_mode = s[1]
                match (escape_mode):
                    case "n":
                        path += '\n'
                    case "b":
                        path += '\b'
                    case "f":
                        path += '\f'
                    case "t":
                        path += '\t'
                    case "\n":
                        pass
                    case _:
                        print(f"Escape mode \\{escape_mode} is invalid.")
                        return False
                node = node[escape_mode]
                ch = s[2]
                s = s[2:]
                raw_length += 2

        return (path, raw_length) if "" in node else False

    def matched(self, s:str):
        if self.match_long(s) != False:
            return True


class IntConstPattern(DFA):
    # int constant: [0-9]+
    def __init__(self):
        tree = {}
        accept_state: Dict[str, Union[dict, bool]] = {"": True}
        for ch in "0123456789":
            tree[ch] = accept_state
            accept_state[ch] = accept_state
        super().__init__(tree)

class CommentPattern(DFA):
    # int constant: [0-9]+
    def __init__(self):
        accept_state: Dict[str, Union[dict, bool]] = {"": True}
        one_line_state: Dict[str, Union[dict, bool]] = {"": True}
        multiline_state: Dict[str, Union[dict, bool]] = {"": True}
        multiline_state2: Dict[str, Union[dict, bool]] = {"": True}
        slash_state = {'/': one_line_state, '*': multiline_state}
        tree = {'/': slash_state}

        for i in range(0, 0x100):
            one_line_state[chr(i)] = one_line_state
        one_line_state['\n'] = accept_state

        for i in range(0, 0x100):
            multiline_state[chr(i)] = multiline_state 
        multiline_state['*'] = multiline_state2 

        for i in range(0, 0x100):
            multiline_state2[chr(i)] = multiline_state 
        multiline_state2['/'] = accept_state 

        super().__init__(tree)

class Token:

    @classmethod
    def new(cls, token):
        if not cls.DFA.matched(token):
            return None
        return cls(token)

    def __repr__(self):
        return f"{type(self).__name__} - {repr(self.raw_value)}"

    def __init__(self, raw_value):
        self.raw_value = raw_value

    def __eq__(self, other):
        return type(self) == type(other) and self.raw_value == other.raw_value

class Comment(Token):
    DFA = CommentPattern()

class Keyword(Token):
    DFA = KeywordsPattern(keywords)

class Symbol(Token):
    DFA = SymbolPattern(symbol)

class StrConst(Token):
    DFA = StrConstPattern()

    @classmethod
    def new(cls, token):
        if not cls.DFA.matched(token):
            return None
        return cls(token)

    def __init__(self, raw_value):
        self.raw_value = raw_value[1:-1]

class IntConst(Token):
    DFA = IntConstPattern()

class Identifier(Token):
    DFA = IdentifierPattern()

class Lexer:

    @staticmethod
    def tokenize(s: str):
    
        first = s[0]
        i = 0

        # Split tokens
        tokens = []
        while i < len(s):
            if symbol_result := Comment.DFA.match_long(s[i:]):
                if i > 0:
                    tokens.append(s[0:i])
                s = s[i+len(symbol_result):]
                i = 0
                continue

            if symbol_result := Symbol.DFA.match_long(s[i:]):
                if i > 0:
                    tokens.append(s[0:i])
                tokens.append(symbol_result)
                s = s[i+len(symbol_result):]
                i = 0
                continue

            if s[i] == ' ' or s[i] == '\n':
                if i > 0:
                    tokens.append(s[0:i])
                s = s[i+1:]
                i = 0 
                continue

            if s[i] == '\"':
                symbol_result = StrConst.DFA.match_long(s)
                if symbol_result == False:
                    raise ValueError(f"Expected StrConst but got {s[i:i+10]}...")
                symbol_result, raw_length = symbol_result
                if i > 0:
                    tokens.append(s[0:i])
                tokens.append(symbol_result)
                s = s[i+raw_length:]
                i = 0
                continue

            i += 1

        for idx,token in enumerate(tokens):
            first = token[0]
            match(first):
                case first if first in "1234567890":
                    result = IntConst.new(token)
                    if result is None:
                        raise ValueError(f"Expected IntConst but got {token}")
                case first if first in ["\'", "\""]:
                    result = StrConst(token)
                    if result is None:
                        raise ValueError(f"Expected StrConst but got {token}")
                case first if first in symbol:
                    result = Symbol.new(token)
                    if result is None:
                        raise ValueError(f"Expected Symbol but got {token}")
                case _:
                    result = Keyword.new(token)
                    if result is None:
                        result = Identifier.new(token)
                    if result is None:
                        raise ValueError(f"Expected Keyword or Identifier but got {token}")
            tokens[idx] = result

        return tokens

if __name__ == "__main__":
    dfa = DFA(keywords)
    import sys
    if len(sys.argv) != 2:
        print("[Usage] python jackparser.py <file>")
        exit()
    with open(sys.argv[1]) as f:
        tokens = Lexer.tokenize(f.read())
        print(tokens)
