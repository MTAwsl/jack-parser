from jacklexer import Token, Symbol, Keyword, StrConst, IntConst, Identifier, Lexer
from typing import List

def print_with_tab(n, str, *args, **kwargs):
    return print(f"{'  '*n}{str}", *args, **kwargs)

class TokenReader:
    def __init__(self, tokens):
        self._tokens = tokens
        self.pos = 0

    def read(self):
        self.pos += 1
        return self._tokens[self.pos-1]

    def peek(self, n=0):
        return self._tokens[self.pos+n]

    def remaining(self):
        return len(self._tokens) - self.pos

class ParseNode:
    def compile(self):
        raise NotImplementedError

class ClassNode(ParseNode):
    @staticmethod
    def firsttoken(token):
        return token in [Keyword("class")]

    def __init__(self, tokens: TokenReader):
        class_kw = tokens.read()
        class_name = tokens.read()

        assert class_kw == Keyword("class")
        assert tokens.read() == Symbol('{')

        self.name = class_name
        self.vars = []
        self.subroutines = []

        while ClassVarDec.firsttoken(tokens.peek()):
            child = ClassVarDec(tokens)
            self.vars.append(child)

        while SubroutineDec.firsttoken(tokens.peek()):
            child = SubroutineDec(tokens)
            self.subroutines.append(child)

        assert tokens.read() == Symbol('}')

    def print(self, n=0):
        print_with_tab(n, "<class>")
        print_with_tab(n+1, f"<name> {self.name.raw_value} </name>")
        for child in self.vars:
            child.print(n+1)
        for child in self.subroutines:
            child.print(n+1)
        print_with_tab(n, "</class>")

class ClassVarDec(ParseNode):
    @staticmethod
    def firsttoken(token):
        return token in [Keyword("static"), Keyword("field")]

    def __init__(self, tokens: TokenReader):
        self.decl_type = tokens.read()
        data_type = tokens.read()
        assert ClassVarDec.firsttoken(self.decl_type)

        self.vars = [VarNode(data_type, tokens.read())]

        while tokens.peek() == Symbol(','):
            tokens.read()
            self.vars.append(VarNode(data_type, tokens.read()))

        assert tokens.read() == Symbol(";")

    def print(self, n=0):
        print_with_tab(n, "<classvar>")
        print_with_tab(n+1, f"<type> {self.decl_type.raw_value} </type>")
        for child in self.vars:
            child.print(n+1)
        print_with_tab(n, f"</classvar>")

class VarNode(ParseNode):
    def __init__(self, typ, name):
        assert type(typ) in [Keyword, Identifier]
        assert type(typ) != Keyword or typ.raw_value in ["int", "char", "boolean"]
        assert type(name) == Identifier
        self.type = typ
        self.name = name

    def print(self, n=0):
        print_with_tab(n, f"<var>")
        print_with_tab(n+1, f"<type> {self.type.raw_value} </type>")
        print_with_tab(n+1, f"<name> {self.name.raw_value} </name>")
        print_with_tab(n, f"</var>")

class SubroutineDec(ParseNode):
    @staticmethod
    def firsttoken(token):
        return type(token) == Keyword and token.raw_value in ["constructor", "function", "method"]

    def __init__(self, tokens: TokenReader):
        self.func_type = tokens.read()
        self.retn_type = tokens.read()
        self.func_name = tokens.read()
        assert tokens.read() == Symbol('(')
        self.param_list = []

        if tokens.peek() != Symbol(')'):
            self.param_list.append(VarNode(tokens.read(), tokens.read()))
            while tokens.peek() == Symbol(','):
                tokens.read()
                self.param_list.append(VarNode(tokens.read(), tokens.read()))

        assert tokens.read() == Symbol(')')
        self.func_body = SubroutineBody(tokens)
    
    def print(self, n=0):
        print_with_tab(n, "<subroutine>")
        print_with_tab(n+1, f"<func_type> {self.func_type.raw_value} </func_type>")
        print_with_tab(n+1, f"<retn_type> {self.retn_type.raw_value} </retn_type>")
        print_with_tab(n+1, f"<func_name> {self.func_name.raw_value} </func_name>")
        print_with_tab(n+1, f"<params>")
        for param in self.param_list:
            param.print(n+2)
        print_with_tab(n+1, f"</params>")
        self.func_body.print(n+1)
        print_with_tab(n, "</subroutine>")

class SubroutineBody(ParseNode):
    @staticmethod
    def firsttoken(token):
        return token == Symbol("{")

    def __init__(self, tokens: TokenReader):
        assert tokens.read() == Symbol("{")

        self.vars = []

        while tokens.peek() == Keyword('var'):
            tokens.read()

            var_type = tokens.read()

            self.vars.append(VarNode(var_type, tokens.read()))

            while tokens.peek() == Symbol(','):
                tokens.read()
                self.vars.append(VarNode(var_type, tokens.read()))

            assert tokens.read() == Symbol(';')

        self.statements = StatementsNode(tokens)
        assert tokens.read() == Symbol('}')

    def print(self, n=0):
        print_with_tab(n, f"<subroutinebody>")
        print_with_tab(n+1, f"<subroutine_vars>")
        for var in self.vars:
            var.print(n+2)
        print_with_tab(n+1, f"</subroutine_vars>")
        self.statements.print(n+1)
        print_with_tab(n, f"</subroutinebody>")

class StatementsNode(ParseNode):
    
    FOLLOW_TOKEN = Symbol('}')
    LET_TOKEN = Keyword("let")
    IF_TOKEN = Keyword("if")
    WHILE_TOKEN = Keyword("while")
    DO_TOKEN = Keyword("do")
    RETURN_TOKEN = Keyword("return")

    @staticmethod
    def firsttoken(token):
        return type(token) == Keyword and token.raw_value in ["let", "if", "while", "do", "return"]
    
    def __init__(self, tokens: TokenReader):
        self.statements = []

        while True:
            match(tokens.peek()):
                case StatementsNode.LET_TOKEN:
                    self.statements.append(LetNode(tokens))
                case StatementsNode.IF_TOKEN:
                    self.statements.append(IfNode(tokens))
                case StatementsNode.WHILE_TOKEN:
                    self.statements.append(WhileNode(tokens))
                case StatementsNode.DO_TOKEN:
                    self.statements.append(DoNode(tokens))
                case StatementsNode.RETURN_TOKEN:
                    self.statements.append(ReturnNode(tokens))
                case StatementsNode.FOLLOW_TOKEN:
                    break

    def print(self, n=0):
        print_with_tab(n, "<statements>")
        for s in self.statements:
            s.print(n+1)
        print_with_tab(n, "</statements>")
            
class LetNode(ParseNode):
    @staticmethod
    def firsttoken(token):
        return token == Keyword("let")

    def __init__(self, tokens: TokenReader):
        assert tokens.read() == Keyword("let")

        self.var_name = tokens.read()
        assert type(self.var_name) == Identifier

        self.getitem_expr = None

        if tokens.peek() == Symbol('['):
            tokens.read()
            self.getitem_expr = ExpressionNode(tokens)
            assert tokens.read() == Symbol(']')

        assert tokens.read() == Symbol('=')

        self.assign_expr = ExpressionNode(tokens)
        assert tokens.read() == Symbol(';')

    def print(self, n=0):
        print_with_tab(n, "<let_statement>")
        print_with_tab(n+1, f"<name> {self.var_name.raw_value} </name>")
        if self.getitem_expr:
            print_with_tab(n+1, f"<item_expr>")
            self.getitem_expr.print(n+2)
            print_with_tab(n+1, f"</item_expr>")
        print_with_tab(n+1, f"<assign>")
        self.assign_expr.print(n+2)
        print_with_tab(n+1, f"</assign>")
        print_with_tab(n, "</let_statement>")

class IfNode(ParseNode):
    @staticmethod
    def firsttoken(token):
        return token == Keyword("if")

    def __init__(self, tokens: TokenReader):
        assert tokens.read() == Keyword("if")
        assert tokens.read() == Symbol("(")

        self.cond_expr = ExpressionNode(tokens)
        self.else_expr = None
        assert tokens.read() == Symbol(")")
        assert tokens.read() == Symbol("{")
        self.statements = StatementsNode(tokens)
        assert tokens.read() == Symbol("}")

        if tokens.peek() == Keyword("else"):
            tokens.read()
            assert tokens.read() == Symbol("{")
            self.else_expr = StatementsNode(tokens)
            assert tokens.read() == Symbol("}")

    def print(self, n=0):
        print_with_tab(n, "<if_statement>")
        print_with_tab(n+1, "<cond_expr>")
        self.cond_expr.print(n+2)
        print_with_tab(n+1, "</cond_expr>")
        print_with_tab(n+1, "<body>")
        self.statements.print(n+2)
        print_with_tab(n+1, "</body>")
        if self.else_expr:
            print_with_tab(n+1, "<else>")
            self.else_expr.print(n+2)
            print_with_tab(n+1, "</else>")
        print_with_tab(n, "</if_statement>")

class WhileNode(ParseNode):
    @staticmethod
    def firsttoken(token):
        return token == Keyword("while")

    def __init__(self, tokens: TokenReader):
        assert tokens.read() == Keyword("while")
        assert tokens.read() == Symbol("(")

        self.cond_expr = ExpressionNode(tokens)
        assert tokens.read() == Symbol(")")
        assert tokens.read() == Symbol("{")
        self.statements = StatementsNode(tokens)
        assert tokens.read() == Symbol("}")

    def print(self, n=0):
        print_with_tab(n, "<while_statement>")
        print_with_tab(n+1, "<cond_expr>")
        self.cond_expr.print(n+2)
        print_with_tab(n+1, "</cond_expr>")
        print_with_tab(n+1, "<body>")
        self.statements.print(n+2)
        print_with_tab(n+1, "</body>")
        print_with_tab(n, "</while_statement>")


class DoNode(ParseNode):
    @staticmethod
    def firsttoken(token):
        return token == Keyword("do")

    def __init__(self, tokens: TokenReader):
        assert tokens.read() == Keyword("do")
        self.call_expr = CallTermNode(tokens)
        assert tokens.read() == Symbol(";")
    
    def print(self, n=0):
        print_with_tab(n, "<do_statement>")
        self.call_expr.print(n+1)
        print_with_tab(n, "</do_statement>")

class ReturnNode(ParseNode):
    @staticmethod
    def firsttoken(token):
        return token == Keyword("return")

    def __init__(self, tokens: TokenReader):
        assert tokens.read() == Keyword("return")
        self.expr = None
        if ExpressionNode.firsttoken(tokens.peek()):
            self.expr = ExpressionNode(tokens)
        assert tokens.read() == Symbol(";")

    def print(self, n=0):
        print_with_tab(n, "<return_statement>")
        if self.expr:
            self.expr.print(n+1)
        print_with_tab(n, "</return_statement>")

class ExpressionNode(ParseNode):
    @staticmethod
    def firsttoken(token):
        return TermNode.firsttoken(token)

    @staticmethod
    def op_first(token):
        return type(token) == Symbol \
                and token.raw_value \
                in set(['+', '-', '*', '/', '&', '|', '>', '<', '='])

    def __init__(self, tokens: TokenReader):
        self.term = TermNode(tokens)
        self.other_terms = []
        while ExpressionNode.op_first(tokens.peek()):
            self.other_terms.append((tokens.read(), TermNode(tokens)))

    def print(self, n=0):
        print_with_tab(n, f"<expression>")
        print_with_tab(n+1, "<term>")
        self.term.print(n+2)
        print_with_tab(n+1, "</term>")
        for op, term in self.other_terms:
            print_with_tab(n+1, f"<op> {op.raw_value} </op>")
            print_with_tab(n+1, "<term>")
            term.print(n+2)
            print_with_tab(n+1, "</term>")
        print_with_tab(n, f"</expression>")
        
class TermNode(ParseNode):
    @staticmethod
    def firsttoken(token):
        childs = [IntTermNode, StrTermNode, KwConstTermNode, \
            VarTermNode, CallTermNode, ExprTermNode, UnaryTermNode]
        return any(cls.firsttoken(token) for cls in childs)

    def __new__(cls, tokens: TokenReader):
        first = tokens.peek()
        second = tokens.peek(1)
        const_nodes = {
            IntConst: IntTermNode, StrConst: StrTermNode,
            Keyword: KwConstTermNode
        }
        if type(first) in const_nodes.keys():
            return const_nodes[type(first)](tokens)
        if type(first) == Symbol:
            match(first.raw_value):
                case '-' | '~':
                    return UnaryTermNode(tokens)
                case '(':
                    return ExprTermNode(tokens)

        assert type(first) == Identifier
        assert type(second) == Symbol

        """
            term -> Identifier
            term -> Identifier [ expr ]
            term -> Identifier ( exprlist )
            term -> Identifier . Identifier ( exprlist )

            term -> Identifier Factored
            Factord -> [ expr ]
            Factord -> ( exprlist ) 
            Factord -> . Identifier ( exprlist )

        """
        # Follow of Term:
        # op ',' ')', ']', ';'
        # Identifier [ expr ] counts as a term
        if second.raw_value in "+-*/&|<>=,)[];":
            return VarTermNode(tokens)
        if second.raw_value in "(.":
            return CallTermNode(tokens)

        assert False

class IntTermNode(ParseNode):
    @staticmethod
    def firsttoken(token):
        return type(token) == IntConst

    def __init__(self, tokens: TokenReader):
        v = tokens.read()
        assert IntTermNode.firsttoken(v)
        self.value = v

    def print(self, n=0):
        print_with_tab(n, f"<int> {self.value.raw_value} </int>")

class StrTermNode(ParseNode):
    @staticmethod
    def firsttoken(token):
        return type(token) == StrConst

    def __init__(self, tokens: TokenReader):
        v = tokens.read()
        assert StrTermNode.firsttoken(v)
        self.value = v

    def print(self, n=0):
        print_with_tab(n, f"<string> {self.value.raw_value} </string>")

class KwConstTermNode(ParseNode):
    @staticmethod
    def firsttoken(token):
        return type(token) == Keyword \
                and token.raw_value in ["true", "false", "null", "this"]

    def __init__(self, tokens: TokenReader):
        v = tokens.read()
        assert KwConstTermNode.firsttoken(v)
        self.value = v

    def print(self, n=0):
        print_with_tab(n, f"<KeywordConst> {self.value.raw_value} </KeywordConst>")

class VarTermNode(ParseNode):
    @staticmethod
    def firsttoken(token):
        return type(token) == Identifier

    def __init__(self, tokens: TokenReader):
        self.value = tokens.read()
        self.expr = None
        if tokens.peek() == Symbol('['):
            tokens.read()
            self.expr = ExpressionNode(tokens)
            assert tokens.read() == Symbol(']')

    def print(self, n=0):
        if self.expr:
            print_with_tab(n, "<VarRef>")
            print_with_tab(n+1, f"<name> {self.value.raw_value} </name>")
            self.expr.print(n+1)
            print_with_tab(n, "</VarRef>")
        else:
            print_with_tab(n, f"<VarRef> {self.value.raw_value} </VarRef>")

class CallTermNode(ParseNode):
    @staticmethod
    def firsttoken(token):
        return type(token) == Identifier

    def __init__(self, tokens: TokenReader):
        second = tokens.peek(1)
        assert type(second) == Symbol
        match second.raw_value:
            case '(':
                self.object_name = None
                self.func_name = tokens.read()
                self.params = []
                assert type(self.func_name) == Identifier
                assert tokens.read() == Symbol('(')
                if tokens.peek() != Symbol(')'):
                    self.params.append(ExpressionNode(tokens))
                    while tokens.peek() == Symbol(','):
                        self.params.append(ExpressionNode(tokens))
                assert tokens.read() == Symbol(')')
            case '.':
                self.object_name = tokens.read()
                assert tokens.read() == Symbol('.')
                self.func_name = tokens.read()
                self.params= []
                assert type(self.func_name) == Identifier
                assert tokens.read() == Symbol('(')
                if tokens.peek() != Symbol(')'):
                    self.params.append(ExpressionNode(tokens))
                    while tokens.peek() == Symbol(','):
                        tokens.read()
                        self.params.append(ExpressionNode(tokens))
                assert tokens.read() == Symbol(')')
            case _:
                assert False

    def print(self, n=0):
        print_with_tab(n, "<CallExpr>")
        if self.object_name:
            print_with_tab(n+1, f"<obj_name> {self.object_name.raw_value} </obj_name>")
        print_with_tab(n+1, f"<func_name> {self.func_name.raw_value} </func_name>")
        print_with_tab(n+1, f"<params>")
        for p in self.params:
            p.print(n+2)
        print_with_tab(n+1, f"</params>")
        print_with_tab(n, "</CallExpr>")

class ExprTermNode(ParseNode):
    @staticmethod
    def firsttoken(token):
        return token == Symbol('(')

    def __init__(self, tokens: TokenReader):
        assert tokens.read() == Symbol('(')
        self.expr = ExpressionNode(tokens)
        assert tokens.read() == Symbol(')')

    def print(self, n=0):
        print_with_tab(n, "<ExprTerm>")
        self.expr.print(n+1)
        print_with_tab(n, "</ExprTerm>")

class UnaryTermNode(ParseNode):
    @staticmethod
    def firsttoken(token):
        return type(token) == Symbol and token.raw_value in ['-', '~', '(']

    def __init__(self, tokens: TokenReader):
        self.op = tokens.read()
        assert UnaryTermNode.firsttoken(self.op)
        self.term = TermNode(tokens)

    def print(self, n=0):
        print_with_tab(n, "<UnaryTerm>")
        print_with_tab(n+1, f"<op> {self.op.raw_value} </op>")
        self.term.print(n+1)
        print_with_tab(n, "</UnaryTerm>")

class ProgramNode(ParseNode):
    def __init__(self, tokens: TokenReader):
        self.classes = []
        while tokens.remaining():
            assert tokens.peek() == Keyword("class")
            self.classes.append(ClassNode(tokens))
    
    def print(self, n=0):
        for c in self.classes:
            c.print(n)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("[Usage] python jackparser.py <file>")
        exit()
    with open(sys.argv[1]) as f:
        tokens = Lexer.tokenize(f.read())
    reader = TokenReader(tokens)
    head = ProgramNode(reader)
    head.print()
