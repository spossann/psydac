#!/usr/bin/python
# -*- coding: utf-8 -*-

import importlib
from collections.abc import Iterable

from sympy import sympify
from sympy import Add as sp_Add, Mul as sp_Mul, Pow as sp_Pow
from sympy import Integral, Symbol, Tuple
from sympy import Lambda
from sympy import Integer as sp_Integer
from sympy import Float as sp_Float, Rational as sp_Rational
from sympy import preorder_traversal

from sympy.simplify.radsimp   import fraction
from sympy.core.compatibility import with_metaclass
from sympy.core.assumptions   import StdFactKB
from sympy.core.singleton     import S
from sympy.core.function      import Function, Application
from sympy.core.function      import Derivative, UndefinedFunction as sp_UndefinedFunction
from sympy.core.function      import _coeff_isneg
from sympy.core.expr          import Expr, AtomicExpr
from sympy.logic.boolalg      import And as sp_And, Not as sp_Not, Or as sp_Or
from sympy.logic.boolalg      import Boolean as sp_Boolean
from sympy.tensor             import Idx, Indexed, IndexedBase

from sympy.matrices.matrices            import MatrixBase
from sympy.matrices.expressions.matexpr import MatrixSymbol, MatrixElement
from sympy.tensor.array.ndim_array      import NDimArray
from sympy.utilities.iterables          import iterable
from sympy.utilities.misc               import filldedent


from .singleton import Singleton
from .basic     import Basic, PyccelAstNode
from .builtins  import Enumerate, Len, List, Map, Range, Zip, PythonTuple, PythonBool, PythonInt
from .datatypes import (datatype, DataType, CustomDataType, NativeSymbol,
                        NativeInteger, NativeBool, NativeReal,
                        NativeComplex, NativeRange, NativeTensor, NativeString,
                        NativeGeneric, NativeTuple, default_precision)

from .numbers        import BooleanTrue, BooleanFalse, Integer as Py_Integer, ImaginaryUnit
from .functionalexpr import GeneratorComprehension as GC
from .functionalexpr import FunctionalFor

from ..errors.errors import Errors
from ..errors.messages import *

errors = Errors()

# TODO [YG, 12.03.2020]: Move non-Python constructs to other modules
# TODO [YG, 12.03.2020]: Rename classes to avoid name clashes in pyccel/ast
# NOTE: commented-out symbols are never used in Pyccel
__all__ = (
    'PyccelOperator',
    'PyccelPow',
    'PyccelAdd',
    'PyccelMinus',
    'PyccelMul',
    'PyccelDiv',
    'PyccelMod',
    'PyccelFloorDiv',
    'PyccelEq',
    'PyccelNe',
    'PyccelLt',
    'PyccelLe',
    'PyccelGt',
    'PyccelGe',
    'PyccelAnd',
    'PyccelOr',
    'PyccelNot',
    'PyccelRShift',
    'PyccelLShift',
    'PyccelBitXor',
    'PyccelBitOr',
    'PyccelBitAnd',
    'PyccelInvert',
    'PyccelAssociativeParenthesis',
    'PyccelUnary',
    'AddOp',
    'AliasAssign',
    'AnnotatedComment',
    'Argument',
    'AsName',
    'Assert',
    'Assign',
    'AugAssign',
    'Block',
    'Break',
    'ClassDef',
    'CodeBlock',
    'Comment',
    'CommentBlock',
    'Concatenate',
    'ConstructorCall',
    'Continue',
    'Declare',
    'Del',
    'DivOp',
    'Dlist',
    'DoConcurrent',
    'DottedName',
    'DottedVariable',
    'EmptyNode',
    'ErrorExit',
    'Eval',
    'Exit',
    'F2PYFunctionDef',
    'For',
    'ForAll',
    'ForIterator',
    'FunctionCall',
    'FunctionDef',
    'GetDefaultFunctionArg',
    'If',
    'IfTernaryOperator',
    'Import',
    'IndexedElement',
    'IndexedVariable',
    'Interface',
    'Is',
    'IsNot',
    'Load',
    'ModOp',
    'Module',
    'MulOp',
    'NativeOp',
    'NewLine',
    'Nil',
    'ParallelBlock',
    'ParallelRange',
    'ParserResult',
    'Pass',
    'Product',
    'Program',
    'PyccelArraySize',
    'PythonFunction',
    'Random',
    'Return',
    'SeparatorComment',
    'Slice',
    'StarredArguments',
    'String',
    'SubOp',
    'Subroutine',
    'SumFunction',
    'SymbolicAssign',
    'SymbolicPrint',
    'SympyFunction',
    'Tensor',
    'Tile',
    'TupleImport',
    'TupleVariable',
    'ValuedArgument',
    'ValuedVariable',
    'Variable',
    'VariableAddress',
    'Void',
    'VoidFunction',
    'While',
    'With',
    '_atomic',
#    'allocatable_like',
    'create_variable',
    'create_incremented_string',
    'extract_subexpressions',
#    'float2int',
    'get_assigned_symbols',
    'get_initial_value',
    'get_iterable_ranges',
    'inline',
    'int2float',
#    'is_simple_assign',
    'local_sympify',
#    'operator',
#    'op_registry',
    'process_shape',
    'subs'
)

#==============================================================================
local_sympify = {
    'N'    : Symbol('N'),
    'S'    : Symbol('S'),
    'zeros': Symbol('zeros'),
    'ones' : Symbol('ones'),
    'Point': Symbol('Point')
}

#==============================================================================
def broadcast(shape_1, shape_2):
    """ This function broadcast two shapes using numpy broadcasting rules """
    a = len(shape_1)
    b = len(shape_2)
    if a>b:
        new_shape_2 = (1,)*(a-b) + tuple(shape_2)
        new_shape_1 = shape_1
    elif b>a:
        new_shape_1 = (1,)*(b-a) + tuple(shape_1)
        new_shape_2 = shape_2
    else:
        new_shape_2 = shape_2
        new_shape_1 = shape_1

    new_shape = []
    for e1,e2 in zip(new_shape_1, new_shape_2):
        if e1 == e2:
            new_shape.append(e1)
        elif e1 == 1:
            new_shape.append(e2)
        elif e2 == 1:
            new_shape.append(e1)
        elif isinstance(e1, PyccelArraySize) and isinstance(e2, PyccelArraySize):
            new_shape.append(e1)
        elif isinstance(e1, PyccelArraySize):
            new_shape.append(e2)
        elif isinstance(e2, PyccelArraySize):
            new_shape.append(e1)
        else:
            msg = 'operands could not be broadcast together with shapes {} {}'
            msg = msg.format(shape_1, shape_2)
            errors.report(msg,severity='fatal')
    return tuple(new_shape)

def handle_precedence(args, my_precedence):
    precedence = [getattr(a, 'precedence', 17) for a in args]

    if min(precedence) <= my_precedence:

        new_args = []

        for i, (a,p) in enumerate(zip(args, precedence)):
            if (p < my_precedence or (p == my_precedence and i != 0)):
                new_args.append(PyccelAssociativeParenthesis(a))
            else:
                new_args.append(a)
        args = tuple(new_args)

    return args

class PyccelBitOperator(Expr, PyccelAstNode):
    _rank = 0
    _shape = ()

    def __init__(self, args):
        if self.stage == 'syntactic':
            self._args = handle_precedence(args, self.precedence)
            return

        max_precision = 0
        for a in args:
            if a.dtype is NativeInteger() or a.dtype is NativeBool():
                max_precision = max(a.precision, max_precision)
            else:
                raise TypeError('unsupported operand type(s): {}'.format(self))
        self._precision = max_precision
    @property
    def precedence(self):
        return self._precedence

class PyccelRShift(PyccelBitOperator):
    _precedence = 11
    _dtype = NativeInteger()
    def __init__(self, *args):
        super(PyccelRShift, self).__init__(args)
        if self.stage == 'syntactic':
            return
        self._args = [PythonInt(a) if a.dtype is NativeBool() else a for a in args]

class PyccelLShift(PyccelBitOperator):
    _precedence = 11
    _dtype = NativeInteger()
    def __init__(self, *args):
        super(PyccelLShift, self).__init__(args)
        if self.stage == 'syntactic':
            return
        self._args = [PythonInt(a) if a.dtype is NativeBool() else a for a in args]

class PyccelBitXor(PyccelBitOperator):
    _precedence = 9
    def __init__(self, *args):
        super(PyccelBitXor, self).__init__(args)
        if self.stage == 'syntactic':
            return
        if all(a.dtype is NativeInteger() for a in args):
            self._dtype = NativeInteger()
        elif all(a.dtype is NativeBool() for a in args):
            self._dtype = NativeBool()
        else:
            self._dtype = NativeInteger()
            self._args = [PythonInt(a) if a.dtype is NativeBool() else a for a in args]

class PyccelBitOr(PyccelBitOperator):
    _precedence = 8
    def __init__(self, *args):
        super(PyccelBitOr, self).__init__(args)
        if self.stage == 'syntactic':
            return
        if all(a.dtype is NativeInteger() for a in args):
            self._dtype = NativeInteger()
        elif all(a.dtype is NativeBool() for a in args):
            self._dtype = NativeBool()
        else:
            self._dtype = NativeInteger()
            self._args = [PythonInt(a) if a.dtype is NativeBool() else a for a in args]

class PyccelBitAnd(PyccelBitOperator):
    _precedence = 10
    def __init__(self, *args):
        super(PyccelBitAnd, self).__init__(args)
        if self.stage == 'syntactic':
            return
        if all(a.dtype is NativeInteger() for a in args):
            self._dtype = NativeInteger()
        elif all(a.dtype is NativeBool() for a in args):
            self._dtype = NativeBool()
        else:
            self._dtype = NativeInteger()
            self._args = [PythonInt(a) if a.dtype is NativeBool() else a for a in args]

class PyccelInvert(PyccelBitOperator):
    _precedence = 14
    _dtype = NativeInteger()
    def __init__(self, *args):
        super(PyccelInvert, self).__init__(args)
        if self.stage == 'syntactic':
            return
        self._args = [PythonInt(a) if a.dtype is NativeBool() else a for a in args]

class PyccelOperator(Expr, PyccelAstNode):

    def __init__(self, *args):

        if self.stage == 'syntactic':
            self._args = handle_precedence(args, self.precedence)
            return
        integers  = [a for a in args if a.dtype is NativeInteger() or a.dtype is NativeBool()]
        reals     = [a for a in args if a.dtype is NativeReal()]
        complexes = [a for a in args if a.dtype is NativeComplex()]
        strs      = [a for a in args if a.dtype is NativeString()]

        if strs:
            self._dtype = NativeString()
            self._rank  = 0
            self._shape = ()
            assert len(integers + reals + complexes) == 0
        else:
            if complexes:
                self._dtype     = NativeComplex()
                self._precision = max(a.precision for a in complexes)
            elif reals:
                self._dtype     = NativeReal()
                self._precision = max(a.precision for a in reals)
            elif integers:
                self._dtype     = NativeInteger()
                self._precision = max(a.precision for a in integers)
            else:
                raise TypeError('cannot determine the type of {}'.format(self))

            ranks  = [a.rank for a in args]
            shapes = [a.shape for a in args]

            if None in ranks:
                self._rank  = None
                self._shape = None
            elif all(sh is not None for tup in shapes for sh in tup):
                if len(args) == 1:
                    shape = args[0].shape
                else:
                    shape = broadcast(args[0].shape, args[1].shape)

                    for a in args[2:]:
                        shape = broadcast(shape, a.shape)

                self._shape = shape
                self._rank  = len(shape)
            else:
                self._rank  = max(a.rank for a in args)
                self._shape = [None]*self._rank

    @property
    def precedence(self):
        return self._precedence

class PyccelPow(PyccelOperator):
    _precedence  = 15
class PyccelAdd(PyccelOperator):
    _precedence = 12
class PyccelMul(PyccelOperator):
    _precedence = 13
class PyccelMinus(PyccelAdd):
    pass
class PyccelDiv(PyccelOperator):
    _precedence = 13
    def __init__(self, *args):

        if self.stage == 'syntactic':
            self._args = handle_precedence(args, self.precedence)
            return

        integers  = [a for a in args if a.dtype is NativeInteger() or a.dtype is NativeBool()]
        reals     = [a for a in args if a.dtype is NativeReal()]
        complexes = [a for a in args if a.dtype is NativeComplex()]
        if complexes:
            self._dtype     = NativeComplex()
            self._precision = max(a.precision for a in complexes)
        elif reals:
            self._dtype     = NativeReal()
            self._precision = max(a.precision for a in reals)
        elif integers:
            self._dtype     = NativeReal()
            self._precision = default_precision['real']

        ranks  = [a.rank for a in args]
        shapes = [a.shape for a in args]

        if None in ranks:
            self._rank  = None
            self._shape = None

        elif all(sh is not None for tup in shapes for sh in tup):
            shape = broadcast(args[0].shape, args[1].shape)

            for a in args[2:]:
                shape = broadcast(shape, a.shape)

            self._shape = shape
            self._rank  = len(shape)
        else:
            self._rank  = max(a.rank for a in args)
            self._shape = [None]*self._rank

class PyccelMod(PyccelOperator):
    _precedence = 13
class PyccelFloorDiv(PyccelOperator):
    _precedence = 13

class PyccelBooleanOperator(Expr, PyccelAstNode):
    _precedence = 7

    def __init__(self, *args):

        if self.stage == 'syntactic':
            self._args = handle_precedence(args, self.precedence)
            return

        self._dtype = NativeBool()
        self._precision = default_precision['bool']

        ranks  = [a.rank for a in args]
        shapes = [a.shape for a in args]

        if None in ranks:
            self._rank  = None
            self._shape = None

        elif all(sh is not None for tup in shapes for sh in tup):
            shape = broadcast(args[0].shape, args[1].shape)
            for a in args[2:]:
                shape = broadcast(shape, a.shape)

            self._shape = shape
            self._rank  = len(shape)
        else:
            self._rank = max(a.rank for a in args)
            self._shape = [None]*self._rank

    @property
    def precedence(self):
        return self._precedence

class PyccelEq(PyccelBooleanOperator):
    pass
class PyccelNe(PyccelBooleanOperator):
    pass
class PyccelLt(PyccelBooleanOperator):
    pass
class PyccelLe(PyccelBooleanOperator):
    pass
class PyccelGt(PyccelBooleanOperator):
    pass
class PyccelGe(PyccelBooleanOperator):
    pass

class PyccelAssociativeParenthesis(Expr, PyccelAstNode):
    _precedence = 18
    def __init__(self, a):
        if self.stage == 'syntactic':
            return
        self._dtype     = a.dtype
        self._rank      = a.rank
        self._precision = a.precision
        self._shape     = a.shape

    @property
    def precedence(self):
        return self._precedence

class PyccelUnary(Expr, PyccelAstNode):
    _precedence = 14

    def __init__(self, a):

        if self.stage == 'syntactic':
            if (getattr(a, 'precedence', 17) <= self.precedence):
                a = PyccelAssociativeParenthesis(a)
                self._args = (a,)
            return

        self._dtype     = a.dtype
        self._rank      = a.rank
        self._precision = a.precision
        self._shape     = a.shape

    @property
    def precedence(self):
        return self._precedence

class PyccelUnarySub(PyccelUnary):
    pass

class PyccelAnd(Expr, PyccelAstNode):
    _dtype = NativeBool()
    _rank  = 0
    _shape = ()
    _precision = default_precision['bool']
    _precedence = 5

    def __init__(self, *args):
        if self.stage == 'syntactic':
            self._args = handle_precedence(args, self.precedence)

    @property
    def precedence(self):
        return self._precedence

class PyccelOr(Expr, PyccelAstNode):
    _dtype = NativeBool()
    _rank  = 0
    _shape = ()
    _precision = default_precision['bool']
    _precedence = 4

    def __init__(self, *args):
        if self.stage == 'syntactic':
            self._args = handle_precedence(args, self.precedence)

    @property
    def precedence(self):
        return self._precedence

class PyccelNot(Expr, PyccelAstNode):
    _dtype = NativeBool()
    _rank  = 0
    _shape = ()
    _precision = default_precision['bool']
    _precedence = 6

    def __init__(self, *args):
        if self.stage == 'syntactic':
            self._args = handle_precedence(args, self.precedence)

    @property
    def precedence(self):
        return self._precedence

class Is(Basic, PyccelAstNode):

    """Represents a is expression in the code.

    Examples
    --------
    >>> from pyccel.ast import Is
    >>> from pyccel.ast import Nil
    >>> from sympy.abc import x
    >>> Is(x, Nil())
    Is(x, None)
    """
    _dtype = NativeBool()
    _rank  = 0
    _shape = ()
    _precision = default_precision['bool']
    _precedence = 7

    def __init__(self, *args):
        if self.stage == 'syntactic':
            self._args = handle_precedence(args, self.precedence)

    @property
    def lhs(self):
        return self._args[0]

    @property
    def rhs(self):
        return self._args[1]

    @property
    def precedence(self):
        return self._precedence


class IsNot(Basic, PyccelAstNode):

    """Represents a is expression in the code.

    Examples
    --------
    >>> from pyccel.ast import IsNot
    >>> from pyccel.ast import Nil
    >>> from sympy.abc import x
    >>> IsNot(x, Nil())
    IsNot(x, None)
    """

    _dtype = NativeBool()
    _rank  = 0
    _shape = ()
    _precision = default_precision['bool']
    _precedence = 7

    def __init__(self, *args):
        if self.stage == 'syntactic':
            self._args = handle_precedence(args, self.precedence)

    @property
    def lhs(self):
        return self._args[0]

    @property
    def rhs(self):
        return self._args[1]

    @property
    def precedence(self):
        return self._precedence


Relational = (PyccelEq,  PyccelNe,  PyccelLt,  PyccelLe,  PyccelGt,  PyccelGe, PyccelAnd, PyccelOr,  PyccelNot, Is, IsNot)
# TODO - add EmptyStmt => empty lines
#      - update code examples
#      - add examples
#      - Function case
#      - AnnotatedComment case
#      - use Tuple after checking the object is iterable:'funcs=Tuple(*funcs)'
#      - add a new Idx that uses Variable instead of Symbol


def subs(expr, new_elements):
    """
    Substitutes old for new in an expression after sympifying args.

    Parameters
    ----------
    new_elements : list of tuples like [(x,2)(y,3)]
    """

    if len(list(new_elements)) == 0:
        return expr
    if isinstance(expr, (list, tuple, Tuple)):
        return [subs(i, new_elements) for i in expr]

    elif isinstance(expr, While):
        test = subs(expr.test, new_elements)
        body = subs(expr.body, new_elements)
        return While(test, body)

    elif isinstance(expr, For):
        target = subs(expr.target, new_elements)
        it = subs(expr.iterable, new_elements)
        target = expr.target
        it = expr.iterable
        body = subs(expr.body, new_elements)
        return For(target, it, body)

    elif isinstance(expr, If):
        args = []
        for block in expr.args:
            test = block[0]
            stmts = block[1]
            t = subs(test, new_elements)
            s = subs(stmts, new_elements)
            args.append((t, s))
        return If(*args)

    elif isinstance(expr, Return):

        for i in new_elements:
            expr = expr.subs(i[0],i[1])
        return expr

    elif isinstance(expr, Assign):
        new_expr = expr.subs(new_elements)
        new_expr.set_fst(expr.fst)
        return new_expr
    elif isinstance(expr, Expr):
        return expr.subs(new_elements)

    else:
        return expr


def allocatable_like(expr, verbose=False):
    """
    finds attributes of an expression

    Parameters
    ----------
    expr: Expr
        a pyccel expression

    verbose: bool
        talk more
    """

    if isinstance(expr, (Variable, IndexedVariable, IndexedElement)):
        return expr
    elif isinstance(expr, str):
        # if the rhs is a string
        return expr
    elif isinstance(expr, Expr):
        args = [expr]
        while args:
            a = args.pop()
            # XXX: This is a hack to support non-Basic args
            if isinstance(a, str):
                continue

            if a.is_Mul:
                if _coeff_isneg(a):
                    if a.args[0] is S.NegativeOne:
                        a = a.as_two_terms()[1]
                    else:
                        a = -a
                (n, d) = fraction(a)
                if n.is_Integer:
                    args.append(d)
                    continue  # won't be -Mul but could be Add
                elif d is not S.One:
                    if not d.is_Integer:
                        args.append(d)
                    args.append(n)
                    continue  # could be -Mul
            elif a.is_Add:
                aargs = list(a.args)
                negs = 0
                for (i, ai) in enumerate(aargs):
                    if _coeff_isneg(ai):
                        negs += 1
                        args.append(-ai)
                    else:
                        args.append(ai)
                continue
            if a.is_Pow and a.exp is S.NegativeOne:
                args.append(a.base)  # won't be -Mul but could be Add
                continue
            if a.is_Mul or a.is_Pow or a.is_Function or isinstance(a,
                    Derivative) or isinstance(a, Integral):

                o = Symbol(a.func.__name__.upper())
            if not a.is_Symbol and not isinstance(a, (IndexedElement,
                    Function)):
                args.extend(a.args)
            if isinstance(a, Function):
                if verbose:
                    print('Functions not yet available')
                return None
            elif isinstance(a, (Variable, IndexedVariable,
                            IndexedElement)):
                return a
            elif a.is_Symbol:
                raise TypeError('Found an unknown symbol {0}'.format(str(a)))
    else:
        raise TypeError('Unexpected type {0}'.format(type(expr)))



def _atomic(e, cls=None,ignore=()):

    """Return atom-like quantities as far as substitution is
    concerned: Functions and DottedVarviables, Variables. we don't
    return atoms that are inside such quantities too
    """

    pot = preorder_traversal(e)
    seen = []
    atoms_ = []
    if cls is None:
        cls = (Application, DottedVariable, Variable,
               IndexedVariable,IndexedElement)

    for p in pot:
        if p in seen or isinstance(p, ignore):
            pot.skip()
            continue
        seen.append(p)
        if isinstance(p, cls):
            pot.skip()
            atoms_.append(p)

    return atoms_



def extract_subexpressions(expr):
    """this function takes an expression and returns a list
      of statements if this expression contains sub expressions that need
      to be evaluated outside of the expression


      Parameters
      ----------
      expr : Add, Mul, Pow, Application

    """

    stmts = []
    cls   = (sp_Add, sp_Mul, sp_Pow, sp_And,
             sp_Or, sp_Eq, sp_Ne, sp_Lt, sp_Gt,
             sp_Le, sp_Ge)

    id_cls = (Symbol, Indexed, IndexedBase,
              DottedVariable, sp_Float, sp_Integer,
              sp_Rational, ImaginaryUnit,sp_Boolean,
              BooleanTrue, BooleanFalse, String,
              ValuedArgument, Nil, List, PythonTuple,
              StarredArguments)

    func_names = ('diag', 'empty', 'zip', 'enumerate')
    #TODO put only imported functions
    def substitute(expr):
        if isinstance(expr, id_cls):
            return expr
        if isinstance(expr, cls):
            args = expr.args
            args = [substitute(arg) for arg in args]
            return expr.func(*args, evaluate=False)
        elif isinstance(expr, Application):
            args = substitute(expr.args)

            if str(expr.func) in func_names:
                var = create_variable(expr)
                expr = expr.func(*args, evaluate=False)
                expr = Assign(var, expr)
                stmts.append(expr)

                return var
            else:
                expr = expr.func(*args, evaluate=False)
                return expr
        elif isinstance(expr, GC):
            stmts.append(expr)
            return expr.lhs
        elif isinstance(expr,IfTernaryOperator):
            var = create_variable(expr)
            new = Assign(var, expr)
            new.set_fst(expr.fst)
            stmts.append(new)
            return var
        elif isinstance(expr, List):
            args = []
            for i in expr:
                args.append(substitute(i))

            return List(*args, sympify=False)

        elif isinstance(expr, (Tuple, tuple, list)):
            args = []

            for i in expr:
                args.append(substitute(i))
            return args

        else:
            raise TypeError('statement {} not supported yet'.format(type(expr)))


    new_expr  = substitute(expr)
    return stmts, new_expr



#def collect_vars(ast):
#    """ collect variables in order to be declared"""
#    #TODO use the namespace to get the declared variables
#    variables = {}
#    def collect(stmt):
#
#        if isinstance(stmt, Variable):
#            if not isinstance(stmt.name, DottedName):
#                variables[stmt.name] = stmt
#        elif isinstance(stmt, (tuple, Tuple, list)):
#            for i in stmt:
#                collect(i)
#        if isinstance(stmt, For):
#            collect(stmt.target)
#            collect(stmt.body)
#        elif isinstance(stmt, FunctionalFor):
#            collect(stmt.lhs)
#            collect(stmt.loops)
#        elif isinstance(stmt, If):
#            collect(stmt.bodies)
#        elif isinstance(stmt, (While, CodeBlock)):
#            collect(stmt.body)
#        elif isinstance(stmt, (Assign, AliasAssign, AugAssign)):
#            collect(stmt.lhs)
#            if isinstance(stmt.rhs, (Linspace, Diag, Where)):
#                collect(stmt.rhs.index)
#
#
#    collect(ast)
#    return variables.values()

def inline(func, args):
        local_vars = func.local_vars
        body = func.body
        body = subs(body, zip(func.arguments, args))
        return Block(str(func.name), local_vars, body)


def int2float(expr):
    return expr

def float2int(expr):
    return expr

def create_incremented_string(forbidden_exprs, prefix = 'Dummy', counter = 1):
    """This function takes a prefix and a counter and uses them to construct
    a new name of the form:
            prefix_counter
    Where counter is formatted to fill 4 characters
    The new name is checked against a list of forbidden expressions. If the
    constructed name is forbidden then the counter is incremented until a valid
    name is found

      Parameters
      ----------
      forbidden_exprs : Set
                        A set of all the values which are not valid solutions to this problem
      prefix          : str
                        The prefix used to begin the string
      counter         : int
                        The expected value of the next name

      Returns
      ----------
      name            : str
                        The incremented string name
      counter         : int
                        The expected value of the next name

    """
    assert(isinstance(forbidden_exprs, set))
    nDigits = 4

    if prefix is None:
        prefix = 'Dummy'

    name_format = "{prefix}_{counter:0="+str(nDigits)+"d}"
    name = name_format.format(prefix=prefix, counter = counter)
    counter += 1
    while name in forbidden_exprs:
        name = name_format.format(prefix=prefix, counter = counter)
        counter += 1

    forbidden_exprs.add(name)

    return name, counter

def create_variable(forbidden_names, prefix = None, counter = 1):
    """This function takes a prefix and a counter and uses them to construct
    a Symbol with a name of the form:
            prefix_counter
    Where counter is formatted to fill 4 characters
    The new name is checked against a list of forbidden expressions. If the
    constructed name is forbidden then the counter is incremented until a valid
    name is found

      Parameters
      ----------
      forbidden_exprs : Set
                        A set of all the values which are not valid solutions to this problem
      prefix          : str
                        The prefix used to begin the string
      counter         : int
                        The expected value of the next name

      Returns
      ----------
      name            : sympy.Symbol
                        A sympy Symbol with the incremented string name
      counter         : int
                        The expected value of the next name

    """

    name, counter = create_incremented_string(forbidden_names, prefix, counter = counter)

    return Symbol(name), counter

class DottedName(Basic):

    """
    Represents a dotted variable.

    Examples
    --------
    >>> from pyccel.ast.core import DottedName
    >>> DottedName('matrix', 'n_rows')
    matrix.n_rows
    >>> DottedName('pyccel', 'stdlib', 'parallel')
    pyccel.stdlib.parallel
    """

    def __new__(cls, *args):
        return Basic.__new__(cls, *args)

    @property
    def name(self):
        return self._args

    def __str__(self):
        return """.""".join(str(n) for n in self.name)

    def _sympystr(self, printer):
        sstr = printer.doprint
        return """.""".join(sstr(n) for n in self.name)


class AsName(Basic):

    """
    Represents a renaming of a variable, used with Import.

    Examples
    --------
    >>> from pyccel.ast.core import AsName
    >>> AsName('old', 'new')
    old as new
    """

    def __new__(cls, name, target):

        # TODO check

        return Basic.__new__(cls, name, target)

    @property
    def name(self):
        return self._args[0]

    @property
    def target(self):
        return self._args[1]

    def _sympystr(self, printer):
        sstr = printer.doprint
        return '{0} as {1}'.format(sstr(self.name), sstr(self.target))

    def __eq__(self, string):
        if isinstance(string, (str, Symbol)):
            return string == self.target
        else:
            return self is string

    def __hash__(self):
        return hash(self.target)


class Dlist(Basic, PyccelAstNode):

    """ this is equivalent to the zeros function of numpy arrays for the python list.

    Parameters
    ----------
    value : Expr
           a sympy expression which represents the initilized value of the list

    shape : the shape of the array
    """

    def __new__(cls, val, length):
        return Basic.__new__(cls, val, length)

    def __init__(self, val, length):
        self._rank = val.rank
        self._shape = tuple(s if i!= 0 else s*length for i,s in enumerate(val.shape))

    @property
    def val(self):
        return self._args[0]

    @property
    def length(self):
        return self._args[1]


class Assign(Basic):

    """Represents variable assignment for code generation.

    Parameters
    ----------
    lhs : Expr
        Sympy object representing the lhs of the expression. These should be
        singular objects, such as one would use in writing code. Notable types
        include Symbol, MatrixSymbol, MatrixElement, and Indexed. Types that
        subclass these types are also supported.

    rhs : Expr
        Sympy object representing the rhs of the expression. This can be any
        type, provided its shape corresponds to that of the lhs. For example,
        a Matrix type can be assigned to MatrixSymbol, but not to Symbol, as
        the dimensions will not align.

    strict: bool
        if True, we do some verifications. In general, this can be more
        complicated and is treated in pyccel.syntax.

    status: None, str
        if lhs is not allocatable, then status is None.
        otherwise, status is {'allocated', 'unallocated'}

    like: None, Variable
        contains the name of the variable from which the lhs will be cloned.

    Examples
    --------
    >>> from sympy import symbols, MatrixSymbol, Matrix
    >>> from pyccel.ast.core import Assign
    >>> x, y, z = symbols('x, y, z')
    >>> Assign(x, y)
    x := y
    >>> Assign(x, 0)
    x := 0
    >>> A = MatrixSymbol('A', 1, 3)
    >>> mat = Matrix([x, y, z]).T
    >>> Assign(A, mat)
    A := Matrix([[x, y, z]])
    >>> Assign(A[0, 1], x)
    A[0, 1] := x

    """

    def __new__(
        cls,
        lhs,
        rhs,
        strict=False,
        status=None,
        like=None,
        ):
        cls._strict = strict

        if strict:
            lhs = sympify(lhs, locals=local_sympify)
            rhs = sympify(rhs, locals=local_sympify)

            # Tuple of things that can be on the lhs of an assignment

            assignable = (Symbol, MatrixSymbol, MatrixElement, Indexed,
                          Idx)

            # if not isinstance(lhs, assignable):
            #    raise TypeError("Cannot assign to lhs of type %s." % type(lhs))
            # Indexed types implement shape, but don't define it until later. This
            # causes issues in assignment validation. For now, matrices are defined
            # as anything with a shape that is not an Indexed

            lhs_is_mat = hasattr(lhs, 'shape') and not isinstance(lhs,
                    Indexed)
            rhs_is_mat = hasattr(rhs, 'shape') and not isinstance(rhs,
                    Indexed)

            # If lhs and rhs have same structure, then this assignment is ok

            if lhs_is_mat:
                if not rhs_is_mat:
                    raise ValueError('Cannot assign a scalar to a matrix.')
                elif lhs.shape != rhs.shape:
                    raise ValueError("Dimensions of lhs and rhs don't align.")
            elif rhs_is_mat and not lhs_is_mat:
                raise ValueError('Cannot assign a matrix to a scalar.')
        return Basic.__new__(cls, lhs, rhs, status, like)

    def _sympystr(self, printer):
        sstr = printer.doprint
        return '{0} := {1}'.format(sstr(self.lhs), sstr(self.rhs))

    @property
    def lhs(self):
        return self._args[0]

    @property
    def rhs(self):
        return self._args[1]

    # TODO : remove

    @property
    def expr(self):
        return self.rhs

    @property
    def status(self):
        return self._args[2]

    @property
    def like(self):
        return self._args[3]

    @property
    def strict(self):
        return self._strict

    @property
    def is_alias(self):
        """Returns True if the assignment is an alias."""

        # TODO to be improved when handling classes

        lhs = self.lhs
        rhs = self.rhs
        cond = isinstance(rhs, Variable) and rhs.rank > 0
        cond = cond or isinstance(rhs, IndexedElement)
        cond = cond or isinstance(rhs, IndexedVariable)
        cond = cond and isinstance(lhs, Symbol)
        cond = cond or isinstance(rhs, Variable) and rhs.is_pointer
        return cond

    @property
    def is_symbolic_alias(self):
        """Returns True if the assignment is a symbolic alias."""

        # TODO to be improved when handling classes

        lhs = self.lhs
        rhs = self.rhs
        if isinstance(lhs, Variable):
            return isinstance(lhs.dtype, NativeSymbol)
        elif isinstance(lhs, Symbol):
            if isinstance(rhs, Range):
                return True
            elif isinstance(rhs, Variable):
                return isinstance(rhs.dtype, NativeSymbol)
            elif isinstance(rhs, Symbol):
                return True

        return False


class CodeBlock(Basic):

    """Represents a list of stmt for code generation.
       we use it when a single statement in python
       produce multiple statement in the targeted language
    """

    def __new__(cls, body):
        ls = []
        for i in body:
            if isinstance(i, CodeBlock):
                ls += i.body
            else:
                ls.append(i)

        return Basic.__new__(cls, ls)

    def __init__(self, body):
        if len(self._args)>0 and isinstance(self._args[-1], (Assign, AugAssign)):
            self.set_fst(ls[-1].fst)

    @property
    def body(self):
        return self._args[0]

    @property
    def lhs(self):
        return self.body[-1].lhs


class AliasAssign(Basic):

    """Represents aliasing for code generation. An alias is any statement of the
    form `lhs := rhs` where

    Parameters
    ----------
    lhs : Symbol
        at this point we don't know yet all information about lhs, this is why a
        Symbol is the appropriate type.

    rhs : Variable, IndexedVariable, IndexedElement
        an assignable variable can be of any rank and any datatype, however its
        shape must be known (not None)

    Examples
    --------
    >>> from sympy import Symbol
    >>> from pyccel.ast.core import AliasAssign
    >>> from pyccel.ast.core import Variable
    >>> n = Variable('int', 'n')
    >>> x = Variable('int', 'x', rank=1, shape=[n])
    >>> y = Symbol('y')
    >>> AliasAssign(y, x)

    """

    def __new__(cls, lhs, rhs):
        if PyccelAstNode.stage == 'semantic':
            if not lhs.is_pointer:
                raise TypeError('lhs must be a pointer')

            if isinstance(rhs, FunctionCall) and not rhs.funcdef.results[0].is_pointer:
                raise TypeError("A pointer cannot point to the address of a temporary variable")

        if isinstance(rhs, Variable):
            rhs = VariableAddress(rhs)

        return Basic.__new__(cls, lhs, rhs)

    def _sympystr(self, printer):
        sstr = printer.doprint
        return '{0} := {1}'.format(sstr(self.lhs), sstr(self.rhs))

    @property
    def lhs(self):
        return self._args[0]

    @property
    def rhs(self):
        return self._args[1]


class SymbolicAssign(Basic):

    """Represents symbolic aliasing for code generation. An alias is any statement of the
    form `lhs := rhs` where

    Parameters
    ----------
    lhs : Symbol

    rhs : Range

    Examples
    --------
    >>> from sympy import Symbol
    >>> from pyccel.ast.core import SymbolicAssign
    >>> from pyccel.ast.core import Range
    >>> r = Range(0, 3)
    >>> y = Symbol('y')
    >>> SymbolicAssign(y, r)

    """

    def __new__(cls, lhs, rhs):
        return Basic.__new__(cls, lhs, rhs)

    def _sympystr(self, printer):
        sstr = printer.doprint
        return '{0} := {1}'.format(sstr(self.lhs), sstr(self.rhs))

    @property
    def lhs(self):
        return self._args[0]

    @property
    def rhs(self):
        return self._args[1]


# The following are defined to be sympy approved nodes. If there is something
# smaller that could be used, that would be preferable. We only use them as
# tokens.

class NativeOp(metaclass=Singleton):

    """Base type for native operands."""

    pass


class AddOp(NativeOp):

    _symbol = '+'


class SubOp(NativeOp):

    _symbol = '-'


class MulOp(NativeOp):

    _symbol = '*'


class DivOp(NativeOp):

    _symbol = '/'


class ModOp(NativeOp):

    _symbol = '%'


op_registry = {
    '+': AddOp(),
    '-': SubOp(),
    '*': MulOp(),
    '/': DivOp(),
    '%': ModOp(),
    }


def operator(op):
    """Returns the operator singleton for the given operator"""

    if op.lower() not in op_registry:
        raise ValueError('Unrecognized operator ' + op)
    return op_registry[op]


class AugAssign(Assign):
    r"""
    Represents augmented variable assignment for code generation.

    Parameters
    ----------
    lhs : Expr
        Sympy object representing the lhs of the expression. These should be
        singular objects, such as one would use in writing code. Notable types
        include Symbol, MatrixSymbol, MatrixElement, and Indexed. Types that
        subclass these types are also supported.

    op : NativeOp
        Operator (+, -, /, \*, %).

    rhs : Expr
        Sympy object representing the rhs of the expression. This can be any
        type, provided its shape corresponds to that of the lhs. For example,
        a Matrix type can be assigned to MatrixSymbol, but not to Symbol, as
        the dimensions will not align.

    strict: bool
        if True, we do some verifications. In general, this can be more
        complicated and is treated in pyccel.syntax.

    status: None, str
        if lhs is not allocatable, then status is None.
        otherwise, status is {'allocated', 'unallocated'}

    like: None, Variable
        contains the name of the variable from which the lhs will be cloned.

    Examples
    --------
    >>> from pyccel.ast.core import Variable
    >>> from pyccel.ast.core import AugAssign
    >>> s = Variable('int', 's')
    >>> t = Variable('int', 't')
    >>> AugAssign(s, '+', 2 * t + 1)
    s += 1 + 2*t
    """

    def __new__(
        cls,
        lhs,
        op,
        rhs,
        strict=False,
        status=None,
        like=None,
        ):
        cls._strict = strict
        if strict:
            lhs = sympify(lhs, locals=local_sympify)
            rhs = sympify(rhs, locals=local_sympify)

            # Tuple of things that can be on the lhs of an assignment

            assignable = (Symbol, MatrixSymbol, MatrixElement, Indexed)
            if not isinstance(lhs, assignable):
                raise TypeError('Cannot assign to lhs of type %s.'
                                % type(lhs))

            # Indexed types implement shape, but don't define it until later. This
            # causes issues in assignment validation. For now, matrices are defined
            # as anything with a shape that is not an Indexed

            lhs_is_mat = hasattr(lhs, 'shape') and not isinstance(lhs,
                    Indexed)
            rhs_is_mat = hasattr(rhs, 'shape') and not isinstance(rhs,
                    Indexed)

            # If lhs and rhs have same structure, then this assignment is ok

            if lhs_is_mat:
                if not rhs_is_mat:
                    raise ValueError('Cannot assign a scalar to a matrix.'
                            )
                elif lhs.shape != rhs.shape:
                    raise ValueError("Dimensions of lhs and rhs don't align."
                            )
            elif rhs_is_mat and not lhs_is_mat:
                raise ValueError('Cannot assign a matrix to a scalar.')

        if isinstance(op, str):
            op = operator(op)
        elif op not in list(op_registry.values()):
            raise TypeError('Unrecognized Operator')

        return Basic.__new__(
            cls,
            lhs,
            op,
            rhs,
            status,
            like,
            )

    def _sympystr(self, printer):
        sstr = printer.doprint
        return '{0} {1}= {2}'.format(sstr(self.lhs), self.op._symbol,
                sstr(self.rhs))

    @property
    def lhs(self):
        return self._args[0]

    @property
    def op(self):
        return self._args[1]

    @property
    def rhs(self):
        return self._args[2]

    @property
    def status(self):
        return self._args[3]

    @property
    def like(self):
        return self._args[4]

    @property
    def strict(self):
        return self._strict


class While(Basic):

    """Represents a 'while' statement in the code.

    Expressions are of the form:
        "while test:
            body..."

    Parameters
    ----------
    test : expression
        test condition given as a sympy expression
    body : sympy expr
        list of statements representing the body of the While statement.

    Examples
    --------
    >>> from sympy import Symbol
    >>> from pyccel.ast.core import Assign, While
    >>> n = Symbol('n')
    >>> While((n>1), [Assign(n,n-1)])
    While(n > 1, (n := n - 1,))
    """

    def __new__(cls, test, body, local_vars=[]):
        test = sympify(test, locals=local_sympify)

        if PyccelAstNode.stage == 'semantic':
            if test.dtype is not NativeBool():
                test = PythonBool(test)

        if iterable(body):
            body = CodeBlock((sympify(i, locals=local_sympify) for i in body))
        elif not isinstance(body,CodeBlock):
            raise TypeError('body must be an iterable or a CodeBlock')
        return Basic.__new__(cls, test, body, local_vars)

    @property
    def test(self):
        return self._args[0]

    @property
    def body(self):
        return self._args[1]

    @property
    def local_vars(self):
        return self._args[2]


class With(Basic):

    """Represents a 'with' statement in the code.

    Expressions are of the form:
        "while test:
            body..."

    Parameters
    ----------
    test : expression
        test condition given as a sympy expression
    body : sympy expr
        list of statements representing the body of the With statement.

    Examples
    --------

    """

    # TODO check prelude and epilog

    def __new__(
        cls,
        test,
        body,
        settings,
        ):
        test = sympify(test, locals=local_sympify)

        if iterable(body):
            body = CodeBlock((sympify(i, locals=local_sympify) for i in body))
        elif not isinstance(body,CodeBlock):
            raise TypeError('body must be an iterable')

        return Basic.__new__(cls, test, body, settings)

    @property
    def test(self):
        return self._args[0]

    @property
    def body(self):
        return self._args[1]

    @property
    def settings(self):
        return self._args[2]

    @property
    def block(self):
        methods = self.test.cls_base.methods
        for i in methods:
            if str(i.name) == '__enter__':
                enter = i
            elif str(i.name) == '__exit__':
                exit = i
        enter = inline(enter,[])
        exit =  inline(exit, [])

        # TODO check if enter is empty or not first

        body = enter.body.body
        body += self.body.body
        body += exit.body.body
        return Block('with', [], body)


class Product(Basic):

    """
    Represents a Product stmt.

    """

    def __new__(cls, *args):
        if not isinstance(args, (tuple, list, Tuple)):
            raise TypeError('args must be an iterable')
        elif len(args) < 2:
            return args[0]
        return Basic.__new__(cls, *args)

    @property
    def elements(self):
        return self._args


class Tile(Range):

    """
    Representes a tile.

    Examples
    --------
    >>> from pyccel.ast.core import Variable
    >>> from pyccel.ast.core import Tile
    >>> from sympy import Symbol
    >>> s = Variable('int', 's')
    >>> e = Symbol('e')
    >>> Tile(s, e, 1)
    Tile(0, n, 1)
    """

    def __new__(cls, start, stop):
        step = 1
        return Range.__new__(cls, start, stop, step)

    @property
    def start(self):
        return self._args[0]

    @property
    def stop(self):
        return self._args[1]

    @property
    def size(self):
        return self.stop - self.start


class ParallelRange(Range):

    """
    Representes a parallel range using OpenMP/OpenACC.

    Examples
    --------
    >>> from pyccel.ast.core import Variable
    """

    pass


# TODO: implement it as an extension of sympy Tensor?

class Tensor(Basic):

    """
    Base class for tensor.

    Examples
    --------
    >>> from pyccel.ast.core import Variable
    >>> from pyccel.ast.core import Range, Tensor
    >>> from sympy import Symbol
    >>> s1 = Variable('int', 's1')
    >>> s2 = Variable('int', 's2')
    >>> e1 = Variable('int', 'e1')
    >>> e2 = Variable('int', 'e2')
    >>> r1 = Range(s1, e1, 1)
    >>> r2 = Range(s2, e2, 1)
    >>> Tensor(r1, r2)
    Tensor(Range(s1, e1, 1), Range(s2, e2, 1), name=tensor)
    """

    def __new__(cls, *args, **kwargs):
        for r in args:
            cond = isinstance(r, Variable) and isinstance(r.dtype,
                    (NativeRange, NativeTensor))
            cond = cond or isinstance(r, (Range, Tensor))

            if not cond:
                raise TypeError('non valid argument, given {0}'.format(type(r)))

        try:
            name = kwargs['name']
        except KeyError:
            name = 'tensor'

        args = list(args) + [name]

        return Basic.__new__(cls, *args)

    @property
    def name(self):
        return self._args[-1]

    @property
    def ranges(self):
        return self._args[:-1]

    @property
    def dim(self):
        return len(self.ranges)

    def _sympystr(self, printer):
        sstr = printer.doprint
        txt = ', '.join(sstr(n) for n in self.ranges)
        txt = 'Tensor({0}, name={1})'.format(txt, sstr(self.name))
        return txt


# TODO add a name to a block?

class Block(Basic):

    """Represents a block in the code. A block consists of the following inputs

    Parameters
    ----------
    variables: list
        list of the variables that appear in the block.

    declarations: list
        list of declarations of the variables that appear in the block.

    body: list
        a list of statements

    Examples
    --------
    >>> from pyccel.ast.core import Variable, Assign, Block
    >>> n = Variable('int', 'n')
    >>> x = Variable('int', 'x')
    >>> Block([n, x], [Assign(x,2.*n + 1.), Assign(n, n + 1)])
    Block([n, x], [x := 1.0 + 2.0*n, n := 1 + n])
    """

    def __new__(
        cls,
        name,
        variables,
        body):
        if not isinstance(name, str):
            raise TypeError('name must be of type str')
        if not iterable(variables):
            raise TypeError('variables must be an iterable')
        for var in variables:
            if not isinstance(var, Variable):
                raise TypeError('Only a Variable instance is allowed.')
        if iterable(body):
            body = CodeBlock(body)
        elif not isinstance(body, CodeBlock):
            raise TypeError('body must be an iterable or a CodeBlock')
        return Basic.__new__(cls, name, variables, body)

    @property
    def name(self):
        return self._args[0]

    @property
    def variables(self):
        return self._args[1]

    @property
    def body(self):
        return self._args[2]

    @property
    def declarations(self):
        return [Declare(i.dtype, i) for i in self.variables]


class ParallelBlock(Block):

    """
    Represents a parallel block in the code.
    In addition to block inputs, there is

    Parameters
    ----------
    clauses: list
        a list of clauses

    Examples
    --------
    >>> from pyccel.ast.core import ParallelBlock
    >>> from pyccel.ast.core import Variable, Assign, Block
    >>> n = Variable('int', 'n')
    >>> x = Variable('int', 'x')
    >>> body = [Assign(x,2.*n + 1.), Assign(n, n + 1)]
    >>> variables = [x,n]
    >>> clauses = []
    >>> ParallelBlock(clauses, variables, body)
    # parallel
    x := 1.0 + 2.0*n
    n := 1 + n
    """

    _prefix = '#'

    def __new__(
        cls,
        clauses,
        variables,
        body,
        ):
        if not iterable(clauses):
            raise TypeError('Expecting an iterable for clauses')

        cls._clauses = clauses

        return Block.__new__(cls, variables, body)

    @property
    def clauses(self):
        return self._clauses

    @property
    def prefix(self):
        return self._prefix

    def _sympystr(self, printer):
        sstr = printer.doprint

        prefix = sstr(self.prefix)
        clauses = ' '.join('{0}'.format(sstr(i)) for i in self.clauses)
        body = '\n'.join('{0}'.format(sstr(i)) for i in self.body)

        code = '{0} parallel {1}\n{2}'.format(prefix, clauses, body)
        return code


class Module(Basic):

    """Represents a module in the code. A block consists of the following inputs

    Parameters
    ----------
    name: str
        name of the module

    variables: list
        list of the variables that appear in the block.

    funcs: list
        a list of FunctionDef instances

    interfaces: list
        a list of Interface instances

    classes: list
        a list of ClassDef instances

    imports: list, tuple
        list of needed imports

    Examples
    --------
    >>> from pyccel.ast.core import Variable, Assign
    >>> from pyccel.ast.core import ClassDef, FunctionDef, Module
    >>> x = Variable('real', 'x')
    >>> y = Variable('real', 'y')
    >>> z = Variable('real', 'z')
    >>> t = Variable('real', 't')
    >>> a = Variable('real', 'a')
    >>> b = Variable('real', 'b')
    >>> body = [Assign(y,x+a)]
    >>> translate = FunctionDef('translate', [x,y,a,b], [z,t], body)
    >>> attributes   = [x,y]
    >>> methods     = [translate]
    >>> Point = ClassDef('Point', attributes, methods)
    >>> incr = FunctionDef('incr', [x], [y], [Assign(y,x+1)])
    >>> decr = FunctionDef('decr', [x], [y], [Assign(y,x-1)])
    >>> Module('my_module', [], [incr, decr], classes = [Point])
    Module(my_module, [], [FunctionDef(), FunctionDef()], [], [ClassDef(Point, (x, y), (FunctionDef(),), [public], (), [], [])], ())
    """

    def __new__(cls, *args, **kwargs):
        return Basic.__new__(cls)

    def __init__(
        self,
        name,
        variables,
        funcs,
        interfaces=[],
        classes=[],
        imports=[],
        ):
        if not isinstance(name, str):
            raise TypeError('name must be a string')

        if not iterable(variables):
            raise TypeError('variables must be an iterable')
        for i in variables:
            if not isinstance(i, Variable):
                raise TypeError('Only a Variable instance is allowed.')

        if not iterable(funcs):
            raise TypeError('funcs must be an iterable')

        for i in funcs:
            if not isinstance(i, FunctionDef):
                raise TypeError('Only a FunctionDef instance is allowed.'
                                )

        if not iterable(classes):
            raise TypeError('classes must be an iterable')
        for i in classes:
            if not isinstance(i, ClassDef):
                raise TypeError('Only a ClassDef instance is allowed.')

        if not iterable(interfaces):
            raise TypeError('interfaces must be an iterable')
        for i in interfaces:
            if not isinstance(i, Interface):
                raise TypeError('Only a Inteface instance is allowed.')

        if not iterable(imports):
            raise TypeError('imports must be an iterable')
        imports = list(imports)
        for i in classes:
            imports += i.imports
        imports = set(imports)  # for unicity
        imports = Tuple(*imports, sympify=False)

        self._name = name
        self._variables = variables
        self._funcs = funcs
        self._interfaces = interfaces
        self._classes = classes
        self._imports = imports

    @property
    def name(self):
        return self._name

    @property
    def variables(self):
        return self._variables

    @property
    def funcs(self):
        return self._funcs

    @property
    def interfaces(self):
        return self._interfaces

    @property
    def classes(self):
        return self._classes

    @property
    def imports(self):
        return self._imports

    @property
    def declarations(self):
        return [Declare(i.dtype, i) for i in self.variables]

    @property
    def body(self):
        return self.interfaces + self.funcs + self.classes

    def set_name(self, new_name):
        self._name = new_name

class Program(Basic):

    """Represents a Program in the code. A block consists of the following inputs

    Parameters
    ----------
    variables: list
        list of the variables that appear in the block.

    declarations: list
        list of declarations of the variables that appear in the block.

    body: list
        a list of statements

    imports: list, tuple
        list of needed imports

    """

    def __new__(
        cls,
        name,
        variables,
        body,
        imports=[],
        ):

        if not isinstance(name, str):
            raise TypeError('name must be a string')

        if not iterable(variables):
            raise TypeError('variables must be an iterable')

        for i in variables:
            if not isinstance(i, Variable):
                raise TypeError('Only a Variable instance is allowed.')

        if not iterable(body):
            raise TypeError('body must be an iterable')
        body = CodeBlock(body)

        if not iterable(imports):
            raise TypeError('imports must be an iterable')

        imports = set(imports)  # for unicity
        imports = Tuple(*imports, sympify=False)

        return Basic.__new__(
            cls,
            name,
            variables,
            body,
            imports,
            )

    @property
    def name(self):
        return self._args[0]

    @property
    def variables(self):
        return self._args[1]

    @property
    def body(self):
        return self._args[2]

    @property
    def imports(self):
        return self._args[3]

    @property
    def declarations(self):
        return [Declare(i.dtype, i) for i in self.variables]


class For(Basic):

    """Represents a 'for-loop' in the code.

    Expressions are of the form:
        "for target in iter:
            body..."

    Parameters
    ----------
    target : symbol
        symbol representing the iterator
    iter : iterable
        iterable object. for the moment only Range is used
    body : sympy expr
        list of statements representing the body of the For statement.

    Examples
    --------
    >>> from sympy import symbols, MatrixSymbol
    >>> from pyccel.ast.core import Assign, For
    >>> i,b,e,s,x = symbols('i,b,e,s,x')
    >>> A = MatrixSymbol('A', 1, 3)
    >>> For(i, (b,e,s), [Assign(x,x-1), Assign(A[0, 1], x)])
    For(i, Range(b, e, s), (x := x - 1, A[0, 1] := x))
    """

    def __new__(
        cls,
        target,
        iter,
        body,
        local_vars = [],
        strict=True,
        ):
        if strict:
            target = sympify(target, locals=local_sympify)

            cond_iter = iterable(iter)
            cond_iter = cond_iter or isinstance(iter, (Range, Product,
                    Enumerate, Zip, Map))
            cond_iter = cond_iter or isinstance(iter, Variable) \
                and is_iterable_datatype(iter.dtype)
          #  cond_iter = cond_iter or isinstance(iter, ConstructorCall) \
          #      and is_iterable_datatype(iter.arguments[0].dtype)
            if not cond_iter:
                raise TypeError('iter must be an iterable')

            if iterable(body):
                body = CodeBlock((sympify(i, locals=local_sympify) for i in
                             body))
            elif not isinstance(body,CodeBlock):
                raise TypeError('body must be an iterable or a Codeblock')

        return Basic.__new__(cls, target, iter, body, local_vars)

    @property
    def target(self):
        return self._args[0]

    @property
    def iterable(self):
        return self._args[1]

    @property
    def body(self):
        return self._args[2]

    @property
    def local_vars(self):
        return self._args[3]

    def insert2body(self, stmt):
        self.body.append(stmt)



class DoConcurrent(For):
    pass


class ForAll(Basic):
    """ class that represents the forall statement in fortran"""
    def __new__(cls, iter, target, mask, body):

        if not isinstance(iter, Range):
            raise TypeError('iter must be of type Range')

        return Basic.__new__(cls, iter, target, mask, body)


    @property
    def iter(self):
        return self._args[0]

    @property
    def target(self):
        return self._args[1]

    @property
    def mask(self):
        return self._args[2]

    @property
    def body(self):
        return self._args[3]

class ForIterator(For):

    """Class that describes iterable classes defined by the user."""

    def __new__(
        cls,
        target,
        iter,
        body,
        strict=True,
        ):

        if isinstance(iter, Symbol):
            iter = Range(Len(iter))
        return For.__new__(cls, target, iter, body, strict)

    # TODO uncomment later when we intriduce iterators
    # @property
    # def target(self):
    #    ts = super(ForIterator, self).target

    #    if not(len(ts) == self.depth):
    #        raise ValueError('wrong number of targets')

    #    return ts

    @property
    def depth(self):
        it = self.iterable
        if isinstance(it, Variable):
            if isinstance(it.dtype, NativeRange):
                return 1
            if isinstance(it.dtype, NativeTensor):

                # TODO must be computed

                return 2

            cls_base = it.cls_base
            if not cls_base:
                raise TypeError('cls_base undefined')

            methods = cls_base.methods_as_dict
            it_method = methods['__iter__']

            it_vars = []
            for stmt in it_method.body:
                if isinstance(stmt, Assign):
                    it_vars.append(stmt.lhs)

            n = len(set([str(var.name) for var in it_vars]))
            return n
        else:

            return 1

    @property
    def ranges(self):
        return get_iterable_ranges(self.iterable)

class ConstructorCall(AtomicExpr):

    """
    It  serves as a constructor for undefined function classes.

    Parameters
    ----------
    func: FunctionDef, str
        an instance of FunctionDef or function name

    arguments: list, tuple, None
        a list of arguments.

    kind: str
        'function' or 'procedure'. default value: 'function'
    """

    is_commutative = True

    # TODO improve

    def __new__(
        cls,
        func,
        arguments,
        cls_variable=None,
        kind='function',
        ):
        if not isinstance(func, (FunctionDef, Interface, str)):
            raise TypeError('Expecting func to be a FunctionDef or str')

        f_name = func.name

        return Basic.__new__(cls, f_name)

    def __init__(
        self,
        func,
        arguments,
        cls_variable=None,
        kind='function',
        ):

        if isinstance(func, FunctionDef):
            kind = func.kind

        self._cls_variable = cls_variable

        self._kind = kind
        self._func = func
        self._arguments = arguments

    def _sympystr(self, printer):
        sstr = printer.doprint
        name = sstr(self.name)
        args = ''
        if not self.arguments is None:
            args = ', '.join(sstr(i) for i in self.arguments)
        return '{0}({1})'.format(name, args)

    @property
    def func(self):
        return self._func

    @property
    def kind(self):
        return self._kind

    @property
    def arguments(self):
        return self._arguments

    @property
    def cls_variable(self):
        return self._cls_variable

    @property
    def name(self):
        if isinstance(self.func, FunctionDef):
            return self.func.name
        else:
            return self.func



class Nil(Basic):

    """
    class for None object in the code.
    """

    def _sympystr(self, printer):
        sstr = printer.doprint
        return sstr('None')


class Void(Basic):

    pass

class VoidFunction(Basic):
    #this class is used in order to eliminate certain atoms
    # in an arithmitic expression so that we dont take them into
    # consideration
    def __new__(*args):
        return Symbol("""x9846548484665
                      494794564465165161561""")

class Variable(Symbol, PyccelAstNode):

    """Represents a typed variable.

    Parameters
    ----------
    dtype : str, DataType
        The type of the variable. Can be either a DataType,
        or a str (bool, int, real).

    name : str, list, DottedName
        The sympy object the variable represents. This can be either a string
        or a dotted name, when using a Class attribute.

    rank : int
        used for arrays. [Default value: 0]

    allocatable: bool
        used for arrays, if we need to allocate memory [Default value: False]

    is_stack_array: bool
        used for arrays, if memory should be allocated on the stack [Default value: False]

    is_pointer: bool
        if object is a pointer [Default value: False]

    is_target: bool
        if object is pointed to by another variable [Default value: False]

    is_polymorphic: bool
        if object can be instance of class or any inherited class [Default value: False]

    is_optional: bool
        if object is an optional argument of a function [Default value: False]

    shape: int or list
        shape of the array. [Default value: None]

    cls_base: class
        class base if variable is an object or an object member [Default value: None]

    order : str
        used for arrays. Indicates whether the data is stored in C or Fortran format in memory [Default value: 'C']

    precision : str
        Precision of the data type [Default value: depends on the datatype]

    is_argument: bool
        if object is the argument of a function [Default value: False]

    is_kwonly: bool
        if object is an argument which can only be specified using its keyword

    Examples
    --------
    >>> from pyccel.ast.core import Variable
    >>> Variable('int', 'n')
    n
    >>> n = 4
    >>> Variable('real', 'x', rank=2, shape=(n,2), allocatable=True)
    x
    >>> Variable('int', DottedName('matrix', 'n_rows'))
    matrix.n_rows
    """

    def __new__( cls, *args, **kwargs ):
        return Basic.__new__(cls)

    def __init__(
        self,
        dtype,
        name,
        *,
        rank=0,
        allocatable=False,
        is_stack_array = False,
        is_pointer=False,
        is_target=False,
        is_polymorphic=None,
        is_optional=False,
        shape=None,
        cls_base=None,
        order='C',
        precision=0,
        is_argument=False,
        is_kwonly=False
        ):

        # ------------ PyccelAstNode Properties ---------------
        if isinstance(dtype, str) or str(dtype) == '*':

            dtype = datatype(str(dtype))
        elif not isinstance(dtype, DataType):
            raise TypeError('datatype must be an instance of DataType.')

        if not isinstance(rank, int):
            raise TypeError('rank must be an instance of int.')

        if rank == 0:
            shape = ()

        if shape is None:
            shape = tuple(None for i in range(rank))

        if not precision:
            if isinstance(dtype, NativeInteger):
                precision = default_precision['int']
            elif isinstance(dtype, NativeReal):
                precision = default_precision['real']
            elif isinstance(dtype, NativeComplex):
                precision = default_precision['complex']
            elif isinstance(dtype, NativeBool):
                precision = default_precision['bool']
        if not isinstance(precision,int) and precision is not None:
            raise TypeError('precision must be an integer or None.')

        self._alloc_shape = shape
        self._dtype = dtype
        self._shape = self.process_shape(shape)
        self._rank  = rank
        self._precision = precision

        # ------------ Variable Properties ---------------
        # if class attribute
        if isinstance(name, str):
            name = name.split(""".""")
            if len(name) == 1:
                name = name[0]
            else:
                name = DottedName(*name)

        if not isinstance(name, (str, DottedName)):
            raise TypeError('Expecting a string or DottedName, given {0}'.format(type(name)))
        self._name = name

        if allocatable is None:
            allocatable = False
        self.allocatable = allocatable

        if is_stack_array is None:
            is_stack_array = False
        elif not isinstance(is_stack_array, bool):
            raise TypeError('is_stack_array must be a boolean.')
        self._is_stack_array = is_stack_array

        if is_pointer is None:
            is_pointer = False
        self.is_pointer = is_pointer

        if is_target is None:
            is_target = False
        self.is_target = is_target

        if is_polymorphic is None:
            if isinstance(dtype, CustomDataType):
                is_polymorphic = dtype.is_polymorphic
            else:
                is_polymorphic = False
        elif not isinstance(is_polymorphic, bool):
            raise TypeError('is_polymorphic must be a boolean.')
        self._is_polymorphic = is_polymorphic

        if is_optional is None:
            is_optional = False
        elif not isinstance(is_optional, bool):
            raise TypeError('is_optional must be a boolean.')
        self._is_optional = is_optional

        self._cls_base       = cls_base
        self._order          = order
        self._is_argument    = is_argument
        self._is_kwonly      = is_kwonly

    def process_shape(self, shape):
        if not hasattr(shape,'__iter__'):
            shape = [shape]

        new_shape = []
        for i,s in enumerate(shape):
            if isinstance(s,(Py_Integer, PyccelArraySize)):
                new_shape.append(s)
            elif isinstance(s, sp_Integer):
                new_shape.append(Py_Integer(s.p))
            elif isinstance(s, int):
                new_shape.append(Py_Integer(s))
            elif s is None or isinstance(s,(Variable, Slice, PyccelAstNode, Function)):
                new_shape.append(PyccelArraySize(self, i))
            else:
                raise TypeError('shape elements cannot be '+str(type(s))+'. They must be one of the following types: Integer(pyccel), Variable, Slice, PyccelAstNode, Integer(sympy), int, Function')
        return tuple(new_shape)

    @property
    def name(self):
        return self._name

    @property
    def alloc_shape(self):
        return self._alloc_shape

    @property
    def allocatable(self):
        return self._allocatable

    @allocatable.setter
    def allocatable(self, allocatable):
        if not isinstance(allocatable, bool):
            raise TypeError('allocatable must be a boolean.')
        self._allocatable = allocatable

    @property
    def cls_base(self):
        return self._cls_base

    @property
    def is_pointer(self):
        return self._is_pointer

    @is_pointer.setter
    def is_pointer(self, is_pointer):
        if not isinstance(is_pointer, bool):
            raise TypeError('is_pointer must be a boolean.')
        self._is_pointer = is_pointer

    @property
    def is_target(self):
        return self._is_target

    @is_target.setter
    def is_target(self, is_target):
        if not isinstance(is_target, bool):
            raise TypeError('is_target must be a boolean.')
        self._is_target = is_target

    @property
    def is_polymorphic(self):
        return self._is_polymorphic

    @property
    def is_optional(self):
        return self._is_optional

    @property
    def order(self):
        return self._order

    @property
    def is_stack_array(self):
        return self._is_stack_array

    @is_stack_array.setter
    def is_stack_array(self, is_stack_array):
        self._is_stack_array = is_stack_array

    @property
    def is_argument(self):
        return self._is_argument

    @property
    def is_kwonly(self):
        return self._is_kwonly

    @property
    def is_ndarray(self):
        """user friendly method to check if the variable is an ndarray:
            1. have a rank > 0
            2. dtype is one among {int, bool, real, complex}
        """

        if self.rank == 0:
            return False
        return isinstance(self.dtype, (NativeInteger, NativeBool,
                          NativeReal, NativeComplex))

    def __str__(self):
        if isinstance(self.name, (str, DottedName)):
            return str(self.name)
        elif self.name is iterable:
            return """.""".join(str(n) for n in self.name)

    def _sympystr(self, printer):
        sstr = printer.doprint
        if isinstance(self.name, (str, DottedName)):
            return '{}'.format(sstr(self.name))
        elif self.name is iterable:
            return """.""".join(sstr(n) for n in self.name)

    def inspect(self):
        """inspects the variable."""

        print('>>> Variable')
        print( '  name           = {}'.format(self.name))
        print( '  dtype          = {}'.format(self.dtype))
        print( '  rank           = {}'.format(self.rank))
        print( '  allocatable    = {}'.format(self.allocatable))
        print( '  shape          = {}'.format(self.shape))
        print( '  cls_base       = {}'.format(self.cls_base))
        print( '  is_pointer     = {}'.format(self.is_pointer))
        print( '  is_target      = {}'.format(self.is_target))
        print( '  is_polymorphic = {}'.format(self.is_polymorphic))
        print( '  is_optional    = {}'.format(self.is_optional))
        print( '<<<')

    def clone(self, name, new_class = None, **kwargs):

        # TODO check it is up to date

        if (new_class is None):
            cls = eval(self.__class__.__name__)
        else:
            cls = new_class

        return cls(
            self.dtype,
            name,
            rank=kwargs.pop('rank',self.rank),
            allocatable=kwargs.pop('allocatable',self.allocatable),
            shape=kwargs.pop('shape',self.shape),
            is_pointer=kwargs.pop('is_pointer',self.is_pointer),
            is_target=kwargs.pop('is_target',self.is_target),
            is_polymorphic=kwargs.pop('is_polymorphic',self.is_polymorphic),
            is_optional=kwargs.pop('is_optional',self.is_optional),
            cls_base=kwargs.pop('cls_base',self.cls_base),
            )

    def __getnewargs__(self):
        """used for Pickling self."""

        args = (
            self.dtype,
            self.name,
            self.rank,
            self.allocatable,
            self.is_pointer,
            self.is_polymorphic,
            self.is_optional,
            self.shape,
            self.cls_base,
            )
        return args

    def _eval_subs(self, old, new):
        return self

    def _eval_is_positive(self):
        #we do this inorder to infere the type of Pow expression correctly
        return self.is_real


class DottedVariable(AtomicExpr, sp_Boolean, PyccelAstNode):

    """
    Represents a dotted variable.
    """

    def __new__(cls, lhs, rhs):

        if not isinstance(lhs, (
            Variable,
            Symbol,
            IndexedVariable,
            IndexedElement,
            IndexedBase,
            Indexed,
            Function,
            DottedVariable,
            )):
            raise TypeError('Expecting a Variable or a function call, got instead {0} of type {1}'.format(str(lhs),
                            type(lhs)))

        if not isinstance(rhs, (
            Variable,
            Symbol,
            IndexedVariable,
            IndexedElement,
            IndexedBase,
            Indexed,
            FunctionCall,
            Function,
            )):
            raise TypeError('Expecting a Variable or a function call, got instead {0} of type {1}'.format(str(rhs),
                            type(rhs)))

        return Basic.__new__(cls, lhs, rhs)

    def __init__(self, lhs, rhs):
        if self.stage == 'syntactic':
            return
        self._dtype     = rhs.dtype
        self._rank      = rhs.rank
        self._precision = rhs.precision
        self._shape     = rhs.shape

    @property
    def lhs(self):
        return self._args[0]

    @property
    def rhs(self):
        return self._args[1]

    @property
    def allocatable(self):
        return self._args[1].allocatable

    @allocatable.setter
    def allocatable(self, allocatable):
        self._args[1].allocatable = allocatable

    @property
    def is_pointer(self):
        return self._args[1].is_pointer

    @is_pointer.setter
    def is_pointer(self, is_pointer):
        self._args[1].is_pointer = is_pointer

    @property
    def is_target(self):
        return self._args[1].is_target

    @is_target.setter
    def is_target(self, is_target):
        self._args[1].is_target = is_target

    @property
    def name(self):
        if isinstance(self.lhs, DottedVariable):
            name_0 = self.lhs.name
        else:
            name_0 = str(self.lhs)
        if isinstance(self.rhs, Function):
            name_1 = str(type(self.rhs).__name__)
        elif isinstance(self.rhs, Symbol):
            name_1 = self.rhs.name
        else:
            name_1 = str(self.rhs)
        return name_0 + """.""" + name_1

    def __str__(self):
        return self.name

    def _sympystr(self, Printer):
        return self.name

    @property
    def cls_base(self):
        return self._args[1].cls_base

    @property
    def names(self):
        """Return list of names as strings."""

        ls = []
        for i in [self.lhs, self.rhs]:
            if not isinstance(i, DottedVariable):
                ls.append(str(i))
            else:
                ls += i.names
        return ls

    def _eval_subs(self, old, new):
        return self

    def inspect(self):
        self._args[1].inspect()



class ValuedVariable(Variable):

    """Represents a valued variable in the code.

    Parameters
    ----------
    variable: Variable
        A single variable
    value: Variable, or instance of Native types
        value associated to the variable

    Examples
    --------
    >>> from pyccel.ast.core import ValuedVariable
    >>> n  = ValuedVariable('int', 'n', value=4)
    >>> n
    n := 4
    """

    def __new__(cls, *args, **kwargs):

        # we remove value from kwargs,
        # since it is not a valid argument for Variable

        kwargs.pop('value', Nil())

        return Variable.__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):

        # if value is not given, we set it to Nil
        self._value = kwargs.pop('value', Nil())
        Variable.__init__(self, *args, **kwargs)

    @property
    def value(self):
        return self._value

    def _sympystr(self, printer):
        sstr = printer.doprint

        name = sstr(self.name)
        value = sstr(self.value)
        return '{0}={1}'.format(name, value)

class TupleVariable(Variable):

    """Represents a tuple variable in the code.

    Parameters
    ----------
    arg_vars: Iterable
        Multiple variables contained within the tuple

    Examples
    --------
    >>> from pyccel.ast.core import TupleVariable, Variable
    >>> v1 = Variable('int','v1')
    >>> v2 = Variable('bool','v2')
    >>> n  = TupleVariable([v1, v2],'n')
    >>> n
    n
    """

    def __new__(cls, arg_vars, dtype, name, *args, **kwargs):

        # if value is not given, we set it to Nil
        # we also remove value from kwargs,
        # since it is not a valid argument for Variable

        return Variable.__new__(cls, dtype, name, *args, **kwargs)

    def __init__(self, arg_vars, dtype, name, *args, **kwargs):
        self._vars = tuple(arg_vars)
        self._inconsistent_shape = not all(arg_vars[0].shape==a.shape   for a in arg_vars[1:])
        self._is_homogeneous = not dtype is NativeGeneric()
        Variable.__init__(self, dtype, name, *args, **kwargs)

    def get_vars(self):
        if self._is_homogeneous:
            indexed_var = IndexedVariable(self, dtype=self.dtype, shape=self.shape,
                prec=self.precision, order=self.order, rank=self. rank)
            args = [Slice(None,None)]*(self.rank-1)
            return [indexed_var[args + [i]] for i in range(len(self._vars))]
        else:
            return self._vars

    def get_var(self, variable_idx):
        return self._vars[variable_idx]

    def rename_var(self, variable_idx, new_name):
        self._vars[variable_idx] = self._vars[variable_idx].clone(new_name)

    def __getitem__(self,idx):
        return self.get_var(idx)

    def __iter__(self):
        return self._vars.__iter__()

    def __len__(self):
        return len(self._vars)

    @property
    def inconsistent_shape(self):
        return self._inconsistent_shape

    @property
    def is_homogeneous(self):
        return self._is_homogeneous

    @is_homogeneous.setter
    def is_homogeneous(self, is_homogeneous):
        self._is_homogeneous = is_homogeneous

    @Variable.allocatable.setter
    def allocatable(self, allocatable):
        if not isinstance(allocatable, bool):
            raise TypeError('allocatable must be a boolean.')
        self._allocatable = allocatable
        for var in self._vars:
            var.allocatable = allocatable

    @Variable.is_pointer.setter
    def is_pointer(self, is_pointer):
        if not isinstance(is_pointer, bool):
            raise TypeError('is_pointer must be a boolean.')
        self._is_pointer = is_pointer
        for var in self._vars:
            var.is_pointer = is_pointer

    @Variable.is_target.setter
    def is_target(self, is_target):
        if not isinstance(is_target, bool):
            raise TypeError('is_target must be a boolean.')
        self._is_target = is_target
        for var in self._vars:
            var.is_target = is_target

class Constant(ValuedVariable, PyccelAstNode):

    """

    Examples
    --------

    """

    pass


class Argument(Symbol, PyccelAstNode):

    """An abstract Argument data structure.

    Examples
    --------
    >>> from pyccel.ast.core import Argument
    >>> n = Argument('n')
    >>> n
    n
    """

    def __new__(cls, name, *, kwonly=False, **assumptions):
        return Symbol.__new__(cls, name, **assumptions)

    def __init__(self, name, *, kwonly=False, **assumptions):
        self._kwonly = kwonly

    @property
    def is_kwonly(self):
        return self._kwonly


class ValuedArgument(Basic):

    """Represents a valued argument in the code.

    Examples
    --------
    >>> from pyccel.ast.core import ValuedArgument
    >>> n = ValuedArgument('n', 4)
    >>> n
    n=4
    """
    def __new__(cls, *args, **kwargs):
        return Basic.__new__(cls)

    def __init__(self, expr, value, *, kwonly = False):
        if isinstance(expr, str):
            expr = Symbol(expr)

        # TODO should we turn back to Argument

        if not isinstance(expr, Symbol):
            raise TypeError('Expecting an argument')

        self._expr   = expr
        self._value  = value
        self._kwonly = kwonly

    @property
    def argument(self):
        return self._expr

    @property
    def value(self):
        return self._value

    @property
    def name(self):
        return self.argument.name

    @property
    def is_kwonly(self):
        return self._kwonly

    def _sympystr(self, printer):
        sstr = printer.doprint

        argument = sstr(self.argument)
        value = sstr(self.value)
        return '{0}={1}'.format(argument, value)

class VariableAddress(Basic, PyccelAstNode):

    """Represents the address of a variable.
    E.g. In C
    VariableAddress(Variable('int','a'))                     is  &a
    VariableAddress(Variable('int','a', is_pointer=True))    is   a
    """

    def __init__(self, variable):
        if not isinstance(variable, Variable):
            raise TypeError('variable must be a variable')
        self._variable = variable

        self._shape     = variable.shape
        self._rank      = variable.rank
        self._dtype     = variable.dtype
        self._precision = variable.precision
        self._order     = variable.order

    @property
    def variable(self):
        return self._variable

class FunctionCall(Basic, PyccelAstNode):

    """Represents a function call in the code.
    """

    def __new__(cls, func, args, current_function=None):

        # ...
        if not isinstance(func, FunctionDef):
            raise TypeError('> expecting a FunctionDef')

        name = func.name
        # ...
        if isinstance(current_function, DottedName):
            current_function = current_function.name[-1]

        if str(current_function) == str(name):
            func.set_recursive()

        if not isinstance(args, (tuple, list, Tuple)):
            raise TypeError('> expecting an iterable')

        # add the messing argument in the case of optional arguments
        f_args = func.arguments
        if not len(args) == len(f_args):
            f_args_dict = dict((a.name,a) if isinstance(a, ValuedVariable) else (a.name, None) for a in f_args)
            keyword_args = []
            for i,a in enumerate(args):
                if not isinstance(a, ValuedVariable):
                    f_args_dict[f_args[i].name] = a
                else:
                    keyword_args = args[i:]
                    break

            for a in keyword_args:
                f_args_dict[a.name] = a.value

            args = [a.value if isinstance(a, ValuedVariable) else a for a in f_args_dict.values()]

        # Ensure the correct syntax is used for pointers
        args = [VariableAddress(a) if isinstance(a, Variable) and f.is_pointer else a for a, f in zip(args, f_args)]

        args = Tuple(*args, sympify=False)
        # ...

        return Basic.__new__(cls, name, args)

    def __init__(self, func, args, current_function=None):

        if str(current_function) == str(func.name):
            if len(func.results)>0 and not isinstance(func.results[0], PyccelAstNode):
                errors.report(RECURSIVE_RESULTS_REQUIRED, symbol=func, severity="fatal")

        self._funcdef     = func
        self._dtype       = func.results[0].dtype if len(func.results) == 1 else NativeTuple()
        self._rank        = func.results[0].rank if len(func.results) == 1 else None
        self._shape       = func.results[0].shape if len(func.results) == 1 else None
        self._precision   = func.results[0].precision if len(func.results) == 1 else None

    @property
    def func(self):
        return self._args[0]

    @property
    def arguments(self):
        return self._args[1]

    @property
    def funcdef(self):
        return self._funcdef

class Return(Basic):

    """Represents a function return in the code.

    Parameters
    ----------
    expr : sympy expr
        The expression to return.

    stmts :represent assign stmts in the case of expression return
    """

    def __new__(cls, expr, stmt=None):

        if stmt and not isinstance(stmt, (Assign, CodeBlock)):
            raise TypeError('stmt should only be of type Assign')

        if PyccelAstNode.stage == 'semantic':
            if isinstance(expr, list):
                expr = [VariableAddress(e) if isinstance(e, Variable) and e.is_pointer else e for e in expr]
            else:
                if isinstance(expr, Variable) and expr.is_pointer:
                    expr = VariableAddress(expr)

        return Basic.__new__(cls, expr, stmt)

    @property
    def expr(self):
        return self._args[0]

    @property
    def stmt(self):
        return self._args[1]

    def __getnewargs__(self):
        """used for Pickling self."""

        args = (self.expr, self.stmt)
        return args


class Interface(Basic):

    """Represent an Interface"""

    def __new__(
        cls,
        name,
        functions,
        hide=False,
        ):
        if not isinstance(name, str):
            raise TypeError('Expecting an str')
        if not isinstance(functions, list):
            raise TypeError('Expecting a list')
        return Basic.__new__(cls, name, functions, hide)

    @property
    def name(self):
        return self._args[0]

    @property
    def functions(self):
        return self._args[1]

    @property
    def hide(self):
        return self.functions[0].hide or self._args[2]

    @property
    def global_vars(self):
        return self.functions[0].global_vars

    @property
    def cls_name(self):
        return self.functions[0].cls_name

    @property
    def kind(self):
        return self.functions[0].kind

    @property
    def imports(self):
        return self.functions[0].imports

    @property
    def decorators(self):
        return self.functions[0].decorators

    @property
    def is_procedure(self):
        return self.functions[0].is_procedure

    def rename(self, newname):
        return Interface(newname, self.functions)


class FunctionDef(Basic):

    """Represents a function definition.

    Parameters
    ----------
    name : str
        The name of the function.

    arguments : iterable
        The arguments to the function.

    results : iterable
        The direct outputs of the function.

    body : iterable
        The body of the function.

    local_vars : list of Symbols
        These are used internally by the routine.

    global_vars : list of Symbols
        Variables which will not be passed into the function.

    cls_name: str
        Class name if the function is a method of cls_name

    hide: bool
        if True, the function definition will not be generated.

    kind: str
        'function' or 'procedure'. default value: 'function'

    is_pure: bool
        True for a function without side effect

    is_elemental: bool
        True for a function that is elemental

    is_private: bool
        True for a function that is private

    is_static: bool
        True for static functions. Needed for f2py

    imports: list, tuple
        a list of needed imports

    decorators: list, tuple
        a list of proporties

    Examples
    --------
    >>> from pyccel.ast.core import Assign, Variable, FunctionDef
    >>> x = Variable('real', 'x')
    >>> y = Variable('real', 'y')
    >>> args        = [x]
    >>> results     = [y]
    >>> body        = [Assign(y,x+1)]
    >>> FunctionDef('incr', args, results, body)
    FunctionDef(incr, (x,), (y,), [y := 1 + x], [], [], None, False, function)

    One can also use parametrized argument, using ValuedArgument

    >>> from pyccel.ast.core import Variable
    >>> from pyccel.ast.core import Assign
    >>> from pyccel.ast.core import FunctionDef
    >>> from pyccel.ast.core import ValuedArgument
    >>> from pyccel.ast.core import GetDefaultFunctionArg
    >>> n = ValuedArgument('n', 4)
    >>> x = Variable('real', 'x')
    >>> y = Variable('real', 'y')
    >>> args        = [x, n]
    >>> results     = [y]
    >>> body        = [Assign(y,x+n)]
    >>> FunctionDef('incr', args, results, body)
    FunctionDef(incr, (x, n=4), (y,), [y := 1 + x], [], [], None, False, function, [])
    """

    def __new__(
        cls,
        name,
        arguments,
        results,
        body,
        *args,
        **kwargs
        ):
        return Basic.__new__(cls)

    def __init__(
        self,
        name,
        arguments,
        results,
        body,
        local_vars=[],
        global_vars=[],
        cls_name=None,
        hide=False,
        kind='function',
        is_static=False,
        imports=[],
        decorators={},
        header=None,
        is_recursive=False,
        is_pure=False,
        is_elemental=False,
        is_private=False,
        is_header=False,
        arguments_inout=[],
        functions = []):

        if isinstance(name, str):
            name = Symbol(name)
        elif isinstance(name, (tuple, list)):
            name_ = []
            for i in name:
                if isinstance(i, str):
                    name = name + Symbol(i)
                elif not isinstance(i, Symbol):
                    raise TypeError('Function name must be Symbol or string'
                                    )
            name = tuple(name_)
        elif not isinstance(name, Symbol):

            raise TypeError('Function name must be Symbol or string')

        # arguments

        # Removed by JO in 06/23 in PR #303: This TypeError was raised upon Sphinx trying to import several of Psydac's modules.
        #if not iterable(arguments):
        #    raise TypeError('arguments must be an iterable')

        # TODO improve and uncomment
#        if not all(isinstance(a, Argument) for a in arguments):
#            raise TypeError("All arguments must be of type Argument")

        arguments = Tuple(*arguments, sympify=False)

        # body

        if iterable(body):
            body = CodeBlock(body)
        elif not isinstance(body,CodeBlock):
            raise TypeError('body must be an iterable or a CodeBlock')

#        body = Tuple(*(i for i in body))
        # results

        if not iterable(results):
            raise TypeError('results must be an iterable')
        results = Tuple(*results, sympify=False)

        # if method

        if cls_name:

            if not isinstance(cls_name, str):
                raise TypeError('cls_name must be a string')

            # if not cls_variable:
             #   raise TypeError('Expecting a instance of {0}'.format(cls_name))

        if kind is None:
            kind = 'function'

        if not isinstance(kind, str):
            raise TypeError('Expecting a string for kind.')

        if not isinstance(is_static, bool):
            raise TypeError('Expecting a boolean for is_static attribute')

        if not kind in ['function', 'procedure']:
            raise ValueError("kind must be one among {'function', 'procedure'}")

        if not iterable(imports):
            raise TypeError('imports must be an iterable')

        if not isinstance(decorators, dict):
            raise TypeError('decorators must be a dict')

        if not isinstance(is_pure, bool):
            raise TypeError('Expecting a boolean for pure')

        if not isinstance(is_elemental, bool):
            raise TypeError('Expecting a boolean for elemental')

        if not isinstance(is_private, bool):
            raise TypeError('Expecting a boolean for private')

        if not isinstance(is_header, bool):
            raise TypeError('Expecting a boolean for header')

        if arguments_inout:
            if not isinstance(arguments_inout, (list, tuple, Tuple)):
                raise TypeError('Expecting an iterable ')

            if not all([isinstance(i, bool) for i in arguments_inout]):
                raise ValueError('Expecting booleans')

        else:
            # TODO shall we keep this?
            arguments_inout = [False for a in arguments]

        if functions:
            for i in functions:
                if not isinstance(i, FunctionDef):
                    raise TypeError('Expecting a FunctionDef')

        self._name            = name
        self._arguments       = arguments
        self._results         = results
        self._body            = body
        self._local_vars      = local_vars
        self._global_vars     = global_vars
        self._cls_name        = cls_name
        self._hide            = hide
        self._kind            = kind
        self._is_static       = is_static
        self._imports         = imports
        self._decorators      = decorators
        self._header          = header
        self._is_recursive    = is_recursive
        self._is_pure         = is_pure
        self._is_elemental    = is_elemental
        self._is_private      = is_private
        self._is_header       = is_header
        self._arguments_inout = arguments_inout
        self._functions       = functions

    @property
    def name(self):
        return self._name

    @property
    def arguments(self):
        return self._arguments

    @property
    def results(self):
        return self._results

    @property
    def body(self):
        return self._body

    @property
    def local_vars(self):
        return self._local_vars

    @property
    def global_vars(self):
        return self._global_vars

    @property
    def cls_name(self):
        return self._cls_name

    @property
    def hide(self):
        return self._hide

    @property
    def kind(self):
        return self._kind

    @property
    def is_static(self):
        return self._is_static

    @property
    def imports(self):
        return self._imports

    @property
    def decorators(self):
        return self._decorators

    @property
    def header(self):
        return self._header

    @property
    def is_recursive(self):
        return self._is_recursive

    @property
    def is_pure(self):
        return self._is_pure

    @property
    def is_elemental(self):
        return self._is_elemental

    @property
    def is_private(self):
        return self._is_private

    @property
    def is_header(self):
        return self._is_header

    @property
    def arguments_inout(self):
        return self._arguments_inout

    @property
    def functions(self):
        return self._functions

    @property
    def doc_string(self):
        return ""

    def print_body(self):
        for s in self.body:
            print(s)

    def set_recursive(self):
        self._is_recursive = True

    def set_cls_name(self, cls_name):
        self._cls_name = cls_name

    def clone(self, newname):
        """
        Create an identical FunctionDef with name
        newname.

        Parameters
        ----------
        newname: str
            new name for the FunctionDef
        """
        args = self.__getnewargs__()
        new_func = FunctionDef(*args)
        new_func.rename(newname)
        return new_func


    def rename(self, newname):
        """
        Rename the FunctionDef name
        newname.

        Parameters
        ----------
        newname: str
            new name for the FunctionDef
        """

        self._name = newname

    def vectorize(self, body , header):
        """ return vectorized FunctionDef """
        decorators = self.decorators
        decorators.pop('vectorize')

        self._name       = 'vec_'+str(self.name)
        self._results    = []
        self._body       = body
        self._kind       = procedure
        self._header     = header
        self._decorators = decorators
        return self

    @property
    def is_procedure(self):
        """Returns True if a procedure."""

        flag = False
        if len(self.results) == 1 and isinstance(self.results[0], Expr):
            vars_ = [i for i in preorder_traversal(self.results)
                     if isinstance(i, Variable)]
            flag = flag or any([i.allocatable or i.rank > 0 for i in
                               vars_])
        else:
            flag = flag or len(self.results) == 1 \
                and self.results[0].allocatable
            flag = flag or len(self.results) == 1 \
                and self.results[0].rank > 0
        flag = flag or len(self.results) > 1
        flag = flag or len(self.results) == 0
        flag = flag or self.kind == 'procedure'
        flag = flag \
            or len(set(self.results).intersection(self.arguments)) > 0
        return flag

    def __getnewargs__(self):
        """used for Pickling self."""

        args = (
                self._name,
                self._arguments,
                self._results,
                self._body,
                self._local_vars,
                self._global_vars,
                self._cls_name,
                self._hide,
                self._kind,
                self._is_static,
                self._imports,
                self._decorators,
                self._header,
                self._is_recursive,
                self._is_pure,
                self._is_elemental,
                self._is_private,
                self._is_header,
                self._arguments_inout,
                self._functions
            )
        return args

    # TODO
    def check_pure(self):
        raise NotImplementedError('')

    # TODO
    def check_elemental(self):
        raise NotImplementedError('')

    def __str__(self):
        result = 'None' if len(self.results) == 0 else \
                    ', '.join(str(r) for r in self.results)
        return '{name}({args}) -> {result}'.format(
                name   = self.name,
                args   = ', '.join(self.args),
                result = result)


class SympyFunction(FunctionDef):

    """Represents a function definition."""

    def rename(self, newname):
        """
        Rename the SympyFunction name by creating a new SympyFunction with
        newname.

        Parameters
        ----------
        newname: str
            new name for the SympyFunction
        """

        return SympyFunction(newname, self.arguments, self.results,
                             self.body, cls_name=self.cls_name)


class PythonFunction(FunctionDef):

    """Represents a Python-Function definition."""

    def rename(self, newname):
        """
        Rename the PythonFunction name by creating a new PythonFunction with
        newname.

        Parameters
        ----------
        newname: str
            new name for the PythonFunction
        """

        return PythonFunction(newname, self.arguments, self.results,
                              self.body, cls_name=self.cls_name)


class F2PYFunctionDef(FunctionDef):
    pass


class GetDefaultFunctionArg(Basic):

    """Creates a FunctionDef for handling optional arguments in the code.

    Parameters
    ----------
    arg: ValuedArgument, ValuedVariable
        argument for which we want to create the function returning the default
        value

    func: FunctionDef
        the function/subroutine in which the optional arg is used

    Examples
    --------
    >>> from pyccel.ast.core import Variable
    >>> from pyccel.ast.core import Assign
    >>> from pyccel.ast.core import FunctionDef
    >>> from pyccel.ast.core import ValuedArgument
    >>> from pyccel.ast.core import GetDefaultFunctionArg
    >>> n = ValuedArgument('n', 4)
    >>> x = Variable('real', 'x')
    >>> y = Variable('real', 'y')
    >>> args        = [x, n]
    >>> results     = [y]
    >>> body        = [Assign(y,x+n)]
    >>> incr = FunctionDef('incr', args, results, body)
    >>> get_n = GetDefaultFunctionArg(n, incr)
    >>> get_n.name
    get_default_incr_n
    >>> get_n
    get_default_incr_n(n=4)

    You can also use **ValuedVariable** as in the following example

    >>> from pyccel.ast.core import ValuedVariable
    >>> n = ValuedVariable('int', 'n', value=4)
    >>> x = Variable('real', 'x')
    >>> y = Variable('real', 'y')
    >>> args        = [x, n]
    >>> results     = [y]
    >>> body        = [Assign(y,x+n)]
    >>> incr = FunctionDef('incr', args, results, body)
    >>> get_n = GetDefaultFunctionArg(n, incr)
    >>> get_n
    get_default_incr_n(n=4)
    """

    def __new__(cls, arg, func):

        if not isinstance(arg, (ValuedArgument, ValuedVariable)):
            raise TypeError('Expecting a ValuedArgument or ValuedVariable'
                            )

        if not isinstance(func, FunctionDef):
            raise TypeError('Expecting a FunctionDef')

        return Basic.__new__(cls, arg, func)

    @property
    def argument(self):
        return self._args[0]

    @property
    def func(self):
        return self._args[1]

    @property
    def name(self):
        text = \
            'get_default_{func}_{arg}'.format(arg=self.argument.name,
                func=self.func.name)
        return text

    def _sympystr(self, printer):
        sstr = printer.doprint

        name = sstr(self.name)
        argument = sstr(self.argument)
        return '{0}({1})'.format(name, argument)


class ClassDef(Basic):

    """Represents a class definition.

    Parameters
    ----------
    name : str
        The name of the class.

    attributes: iterable
        The attributes to the class.

    methods: iterable
        Class methods

    options: list, tuple
        list of options ('public', 'private', 'abstract')

    imports: list, tuple
        list of needed imports

    parent : str
        parent's class name

    Examples
    --------
    >>> from pyccel.ast.core import Variable, Assign
    >>> from pyccel.ast.core import ClassDef, FunctionDef
    >>> x = Variable('real', 'x')
    >>> y = Variable('real', 'y')
    >>> z = Variable('real', 'z')
    >>> t = Variable('real', 't')
    >>> a = Variable('real', 'a')
    >>> b = Variable('real', 'b')
    >>> body = [Assign(y,x+a)]
    >>> translate = FunctionDef('translate', [x,y,a,b], [z,t], body)
    >>> attributes   = [x,y]
    >>> methods     = [translate]
    >>> ClassDef('Point', attributes, methods)
    ClassDef(Point, (x, y), (FunctionDef(translate, (x, y, a, b), (z, t), [y := a + x], [], [], None, False, function),), [public])
    """

    def __new__(
        cls,
        name,
        attributes=[],
        methods=[],
        options=['public'],
        imports=[],
        parent=[],
        interfaces=[],
        ):

        # name

        if isinstance(name, str):
            name = Symbol(name)
        elif not isinstance(name, Symbol):
            raise TypeError('Function name must be Symbol or string')

        # attributes

        if not iterable(attributes):
            raise TypeError('attributes must be an iterable')
        attributes = Tuple(*attributes, sympify=False)

        # methods

        if not iterable(methods):
            raise TypeError('methods must be an iterable')

        # options

        if not iterable(options):
            raise TypeError('options must be an iterable')

        # imports

        if not iterable(imports):
            raise TypeError('imports must be an iterable')

        if not iterable(parent):
            raise TypeError('parent must be iterable')

        if not iterable(interfaces):
            raise TypeError('interfaces must be iterable')

        imports = list(imports)
        for i in methods:
            imports += list(i.imports)

        imports = set(imports)  # for unicity
        imports = Tuple(*imports, sympify=False)

        # ...
        # look if the class has the method __del__
        # d_methods = {}
        # for i in methods:
        #    d_methods[str(i.name).replace('\'','')] = i
        # if not ('__del__' in d_methods):
        #    dtype = DataTypeFactory(str(name), ("_name"), prefix='Custom')
        #    this  = Variable(dtype(), 'self')

            # constructs the __del__ method if not provided
         #   args = []
         #   for a in attributes:
         #       if isinstance(a, Variable):
         #           if a.allocatable:
         #              args.append(a)

         #   args = [Variable(a.dtype, DottedName(str(this), str(a.name))) for a in args]
         #   body = [Del(a) for a in args]

         #   free = FunctionDef('__del__', [this], [], \
         #                      body, local_vars=[], global_vars=[], \
         #                      cls_name='__UNDEFINED__', kind='procedure', imports=[])

         #  methods = list(methods) + [free]
         # TODO move this somewhere else

        methods = Tuple(*methods, sympify=False)

        # ...

        return Basic.__new__(
            cls,
            name,
            attributes,
            methods,
            options,
            imports,
            parent,
            interfaces,
            )

    @property
    def name(self):
        return self._args[0]

    @property
    def attributes(self):
        return self._args[1]

    @property
    def methods(self):
        return self._args[2]

    @property
    def options(self):
        return self._args[3]

    @property
    def imports(self):
        return self._args[4]

    @property
    def parent(self):
        return self._args[5]

    @property
    def interfaces(self):
        return self._args[6]

    @property
    def methods_as_dict(self):
        """Returns a dictionary that contains all methods, where the key is the
        method's name."""

        d_methods = {}
        for i in self.methods:
            d_methods[str(i.name)] = i
        return d_methods

    @property
    def attributes_as_dict(self):
        """Returns a dictionary that contains all attributes, where the key is the
        attribute's name."""

        d_attributes = {}
        for i in self.attributes:
            d_attributes[str(i.name)] = i
        return d_attributes

    # TODO add other attributes?


    def get_attribute(self, O, attr):
        """Returns the attribute attr of the class O of instance self."""

        if not isinstance(attr, str):
            raise TypeError('Expecting attribute to be a string')

        if isinstance(O, Variable):
            cls_name = str(O.name)
        else:
            cls_name = str(O)

        attributes = {}
        for i in self.attributes:
            attributes[str(i.name)] = i

        if not attr in attributes:
            raise ValueError('{0} is not an attribute of {1}'.format(attr,
                             str(self)))

        var = attributes[attr]
        name = DottedName(cls_name, str(var.name))
        return Variable(
            var.dtype,
            name,
            rank=var.rank,
            allocatable=var.allocatable,
            shape=var.shape,
            cls_base=var.cls_base,
            )

    @property
    def is_iterable(self):
        """Returns True if the class has an iterator."""

        names = [str(m.name) for m in self.methods]
        if '__next__' in names and '__iter__' in names:
            return True
        elif '__next__' in names:
            raise ValueError('ClassDef does not contain __iter__ method')
        elif '__iter__' in names:
            raise ValueError('ClassDef does not contain __next__ method')
        else:
            return False

    @property
    def is_with_construct(self):
        """Returns True if the class is a with construct."""

        names = [str(m.name) for m in self.methods]
        if '__enter__' in names and '__exit__' in names:
            return True
        elif '__enter__' in names:
            raise ValueError('ClassDef does not contain __exit__ method')
        elif '__exit__' in names:
            raise ValueError('ClassDef does not contain __enter__ method')
        else:
            return False

    @property
    def hide(self):
        if 'hide' in self.options:
            return True
        else:
            return self.is_iterable or self.is_with_construct

    def _eval_subs(self, old , new):
        return self


class Import(Basic):

    """Represents inclusion of dependencies in the code.

    Parameters
    ----------
    target : str, list, tuple, Tuple
        targets to import

    Examples
    --------
    >>> from pyccel.ast.core import Import
    >>> from pyccel.ast.core import DottedName
    >>> Import('foo')
    import foo

    >>> abc = DottedName('foo', 'bar', 'baz')
    >>> Import(abc)
    import foo.bar.baz

    >>> Import(['foo', abc])
    import foo, foo.bar.baz
    """

    def __new__(cls, source, target = None):

        if not source is None:
            source = Import._format(source)

        return Basic.__new__(cls, source)

    def __init__(self, source, target = None):
        self._target = []
        if isinstance(target, (str, Symbol, DottedName, AsName)):
            self._target = [Import._format(target)]
        elif iterable(target):
            for i in target:
                self._target.append(Import._format(i))

    @staticmethod
    def _format(i):
        if isinstance(i, str):
            if '.' in i:
                return DottedName(*i.split('.'))
            else:
                return Symbol(i)
        if isinstance(i, (DottedName, AsName)):
            return i
        elif isinstance(i, Symbol):
            return i
        else:
            raise TypeError('Expecting a string, Symbol DottedName, given {}'.format(type(i)))

    @property
    def target(self):
        return self._target

    @property
    def source(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        source = sstr(self.source)
        if len(self.target) == 0:
            return 'import {source}'.format(source=source)
        else:
            target = ', '.join([sstr(i) for i in self.target])
            return 'from {source} import {target}'.format(source=source,
                    target=target)

    def define_target(self, new_target):
        self._target.append(new_target)

    def find_module_target(self, new_target):
        for t in self._target:
            if isinstance(t, AsName) and new_target == str(t.name):
                return t.target
            elif new_target == str(t):
                return t
        return None

class TupleImport(Basic):

    def __new__(cls, *args):
        for a in args:
            if not isinstance(a, Import):
                raise TypeError('Expecting an Import statement')
        return Basic.__new__(cls, *args)

    @property
    def imports(self):
        return self._args

    def _sympystr(self, printer):
        sstr = printer.doprint
        return '\n'.join(sstr(n) for n in self.imports)


class Load(Basic):

    """Similar to 'importlib' in python. In addition, we can also provide the
    functions we want to import.

    Parameters
    ----------
    module: str, DottedName
        name of the module to load.

    funcs: str, list, tuple, Tuple
        a string representing the function to load, or a list of strings.

    as_lambda: bool
        load as a Lambda expression, if True

    nargs: int
        number of arguments of the function to load. (default = 1)

    Examples
    --------
    >>> from pyccel.ast.core import Load
    """

    def __new__(
        cls,
        module,
        funcs=None,
        as_lambda=False,
        nargs=1,
        ):
        if not isinstance(module, (str, DottedName, list, tuple,
                          Tuple)):
            raise TypeError('Expecting a string or DottedName, given {0}'.format(type(module)))

        # see syntax

        if isinstance(module, str):
            module = module.replace('__', """.""")

        if isinstance(module, (list, tuple, Tuple)):
            module = DottedName(*module)

        if funcs:
            if not isinstance(funcs, (str, DottedName, list, tuple,
                              Tuple)):
                raise TypeError('Expecting a string or DottedName')

            if isinstance(funcs, str):
                funcs = [funcs]
            elif not isinstance(funcs, (list, tuple, Tuple)):
                raise TypeError('Expecting a string, list, tuple, Tuple')

        if not isinstance(as_lambda, (BooleanTrue, BooleanFalse, bool)):
            raise TypeError('Expecting a boolean, given {0}'.format(as_lambda))

        return Basic.__new__(cls, module, funcs, as_lambda, nargs)

    @property
    def module(self):
        return self._args[0]

    @property
    def funcs(self):
        return self._args[1]

    @property
    def as_lambda(self):
        return self._args[2]

    @property
    def nargs(self):
        return self._args[3]

    def execute(self):
        module = str(self.module)
        package = importlib.import_module(module)

        ls = []
        for f in self.funcs:
            try:
                m = getattr(package, '{0}'.format(str(f)))
            except AttributeError:
                raise ImportError('could not import {0}'.format(f))

            # TODO improve

            if self.as_lambda:
                args = []
                for i in range(0, self.nargs):
                    fi = Symbol('f{0}'.format(i))
                    args.append(fi)
                if len(args) == 1:
                    arg = args[0]
                    m = Lambda(arg, m(arg, evaluate=False))
                else:
                    m = Lambda(tuple(args), m(evaluate=False, *args))

            ls.append(m)

        return ls


# TODO: Should Declare have an optional init value for each var?

class Declare(Basic):

    """Represents a variable declaration in the code.

    Parameters
    ----------
    dtype : DataType
        The type for the declaration.
    variable(s)
        A single variable or an iterable of Variables. If iterable, all
        Variables must be of the same type.
    intent: None, str
        one among {'in', 'out', 'inout'}
    value: Expr
        variable value
    static: bool
        True for a static declaration of an array.

    Examples
    --------
    >>> from pyccel.ast.core import Declare, Variable
    >>> Declare('int', Variable('int', 'n'))
    Declare(NativeInteger(), (n,), None)
    >>> Declare('real', Variable('real', 'x'), intent='out')
    Declare(NativeReal(), (x,), out)
    """

    def __new__(
        cls,
        dtype,
        variable,
        intent=None,
        value=None,
        static=False,
        ):
        if isinstance(dtype, str):
            dtype = datatype(dtype)
        elif not isinstance(dtype, DataType):
            raise TypeError('datatype must be an instance of DataType.')

        if not isinstance(variable, Variable):
            raise TypeError('var must be of type Variable, given {0}'.format(variable))
        if variable.dtype != dtype:
            raise ValueError('All variables must have the same dtype')

        if intent:
            if not intent in ['in', 'out', 'inout']:
                raise ValueError("intent must be one among {'in', 'out', 'inout'}")

        if not isinstance(static, bool):
            raise TypeError('Expecting a boolean for static attribute')

        return Basic.__new__(
            cls,
            dtype,
            variable,
            intent,
            value,
            static,
            )

    @property
    def dtype(self):
        return self._args[0]

    @property
    def variable(self):
        return self._args[1]

    @property
    def intent(self):
        return self._args[2]

    @property
    def value(self):
        return self._args[3]

    @property
    def static(self):
        return self._args[4]


class Subroutine(sp_UndefinedFunction):
    pass


class Break(Basic):

    """Represents a break in the code."""

    pass


class Continue(Basic):

    """Represents a continue in the code."""

    pass


class Raise(Basic):

    """Represents a raise in the code."""

    pass


# TODO: improve with __new__ from Function and add example

class Random(Function, PyccelAstNode):

    """
    Represents a 'random' number in the code.
    """

    # TODO : remove later

    def __str__(self):
        return 'random'

    def __new__(cls, seed):
        return Basic.__new__(cls, seed)

    @property
    def seed(self):
        return self._args[0]


# TODO: improve with __new__ from Function and add example

class SumFunction(Basic, PyccelAstNode):

    """Represents a Sympy Sum Function.

       Parameters
       ----------
       body: Expr
       Sympy Expr in which the sum will be performed.

       iterator:
       a tuple  that containts the index of the sum and it's range.
    """

    def __new__(
        cls,
        body,
        iterator,
        stmts=None,
        ):
        if not isinstance(iterator, (tuple, Tuple)):
            raise TypeError('iterator must be a tuple')
        if not len(iterator) == 3:
            raise ValueError('iterator must be of lenght 3')
        return Basic.__new__(cls, body, iterator, stmts)

    @property
    def body(self):
        return self._args[0]

    @property
    def iterator(self):
        return self._args[1]

    @property
    def stmts(self):
        return self._args[2]


class SymbolicPrint(Basic):

    """Represents a print function of symbolic expressions in the code.

    Parameters
    ----------
    expr : sympy expr
        The expression to return.

    Examples
    --------
    >>> from sympy import symbols
    >>> from pyccel.ast.core import Print
    >>> n,m = symbols('n,m')
    >>> Print(('results', n,m))
    Print((results, n, m))
    """

    def __new__(cls, expr):
        if not iterable(expr):
            raise TypeError('Expecting an iterable')

        for i in expr:
            if not isinstance(i, (Lambda, SymbolicAssign,
                              SympyFunction)):
                raise TypeError('Expecting Lambda, SymbolicAssign, SympyFunction for {}'.format(i))

        return Basic.__new__(cls, expr)

    @property
    def expr(self):
        return self._args[0]


class Del(Basic):

    """Represents a memory deallocation in the code.

    Parameters
    ----------
    variables : list, tuple
        a list of pyccel variables

    Examples
    --------
    >>> from pyccel.ast.core import Del, Variable
    >>> x = Variable('real', 'x', rank=2, shape=(10,2), allocatable=True)
    >>> Del([x])
    Del([x])
    """

    def __new__(cls, expr):

        # TODO: check that the variable is allocatable

        if not iterable(expr):
            expr = Tuple(expr, sympify=False)
        return Basic.__new__(cls, expr)

    @property
    def variables(self):
        return self._args[0]


class EmptyNode(Basic):
    """
    Represents an empty node in the abstract syntax tree (AST).
    When a subtree is removed from the AST, we replace it with an EmptyNode
    object that acts as a placeholder. Using an EmptyNode instead of None
    is more explicit and avoids confusion. Further, finding a None in the AST
    is signal of an internal bug.

    Parameters
    ----------
    text : str
       the comment line

    Examples
    --------
    >>> from pyccel.ast.core import EmptyNode
    >>> EmptyNode()

    """

    def __new__(cls):
        return Basic.__new__(cls)

    def _sympystr(self, printer):
        return ''


class NewLine(Basic):

    """Represents a NewLine in the code.

    Parameters
    ----------
    text : str
       the comment line

    Examples
    --------
    >>> from pyccel.ast.core import NewLine
    >>> NewLine()

    """

    def __new__(cls):
        return Basic.__new__(cls)

    def _sympystr(self, printer):
        return '\n'


class Comment(Basic):

    """Represents a Comment in the code.

    Parameters
    ----------
    text : str
       the comment line

    Examples
    --------
    >>> from pyccel.ast.core import Comment
    >>> Comment('this is a comment')
    # this is a comment
    """

    def __new__(cls, text):
        return Basic.__new__(cls, text)

    @property
    def text(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer._print
        txt  = sstr(self.text)
        if txt.lstrip()[0] != '#':
            txt = '#' + txt
        return txt


class SeparatorComment(Comment):

    """Represents a Separator Comment in the code.

    Parameters
    ----------
    mark : str
        marker

    Examples
    --------
    >>> from pyccel.ast.core import SeparatorComment
    >>> SeparatorComment(n=40)
    # ........................................
    """

    def __new__(cls, n):
        text = """.""" * n
        return Comment.__new__(cls, text)


class AnnotatedComment(Basic):

    """Represents a Annotated Comment in the code.

    Parameters
    ----------
    accel : str
       accelerator id. One among {'omp', 'acc'}

    txt: str
        statement to print

    Examples
    --------
    >>> from pyccel.ast.core import AnnotatedComment
    >>> AnnotatedComment('omp', 'parallel')
    AnnotatedComment(omp, parallel)
    """

    def __new__(cls, accel, txt):
        return Basic.__new__(cls, accel, txt)

    @property
    def accel(self):
        return self._args[0]

    @property
    def txt(self):
        return self._args[1]

    def __getnewargs__(self):
        """used for Pickling self."""

        args = (self.accel, self.txt)
        return args

class CommentBlock(Basic):

    """ Represents a Block of Comments

    Parameters
    ----------
    txt : str

    """
    def __new__(cls, txt):
        if not isinstance(txt, str):
            raise TypeError('txt must be of type str')
        txt = txt.replace('"','')
        txts = txt.split('\n')

        return Basic.__new__(cls, txts)

    @property
    def comments(self):
        return self._args[0]

class IndexedVariable(IndexedBase, PyccelAstNode):

    """
    Represents an indexed variable, like x in x[i], in the code.

    Examples
    --------
    >>> from sympy import symbols, Idx
    >>> from pyccel.ast.core import IndexedVariable
    >>> A = IndexedVariable('A'); A
    A
    >>> type(A)
    <class 'pyccel.ast.core.IndexedVariable'>

    When an IndexedVariable object receives indices, it returns an array with named
    axes, represented by an IndexedElement object:

    >>> i, j = symbols('i j', integer=True)
    >>> A[i, j, 2]
    A[i, j, 2]
    >>> type(A[i, j, 2])
    <class 'pyccel.ast.core.IndexedElement'>

    The IndexedVariable constructor takes an optional shape argument.  If given,
    it overrides any shape information in the indices. (But not the index
    ranges!)

    >>> m, n, o, p = symbols('m n o p', integer=True)
    >>> i = Idx('i', m)
    >>> j = Idx('j', n)
    >>> A[i, j].shape
    (m, n)
    >>> B = IndexedVariable('B', shape=(o, p))
    >>> B[i, j].shape
    (m, n)

    **todo:** fix bug. the last result must be : (o,p)
    """

    def __new__(
        cls,
        label,
        shape=None,
        dtype=None,
        prec=0,
        order=None,
        rank = 0,
        **kw_args
        ):

        if isinstance(label, Application):
            label_name = type(label)
        else:
            label_name = str(label)

        return IndexedBase.__new__(cls, label_name, shape=shape)

    def __init__(
        self,
        label,
        shape=None,
        dtype=None,
        prec=0,
        order=None,
        rank = 0,
        **kw_args
        ):

        if dtype is None:
            raise TypeError('datatype must be provided')
        if isinstance(dtype, str):
            dtype = datatype(dtype)
        elif not isinstance(dtype, DataType):
            raise TypeError('datatype must be an instance of DataType.')


        self._dtype      = dtype
        self._precision  = prec
        self._rank       = rank
        kw_args['order'] = order
        self._kw_args    = kw_args
        self._label      = label

    def __getitem__(self, *args):

        if len(args) == 1 and isinstance(args[0], (Tuple, tuple, list)):
            args = args[0]

        if self.shape and len(self.shape) != len(args):
            raise IndexError('Rank mismatch.')

        obj = IndexedElement(self, *args)
        return obj

    @property
    def order(self):
        return self.kw_args['order']

    @property
    def kw_args(self):
        return self._kw_args

    @property
    def name(self):
        return self._args[0]

    @property
    def internal_variable(self):
        return self._label


    def clone(self, name):
        cls = eval(self.__class__.__name__)
        # TODO what about kw_args in __new__?
        return cls(name, shape=self.shape, dtype=self.dtype,
                   prec=self.precision, order=self.order, rank=self.rank)

    def _eval_subs(self, old, new):
        return self

    def __str__(self):
        return str(self.name)


class IndexedElement(Expr, PyccelAstNode):

    """
    Represents a mathematical object with indices.

    Examples
    --------
    >>> from sympy import symbols, Idx
    >>> from pyccel.ast.core import IndexedVariable
    >>> i, j = symbols('i j', cls=Idx)
    >>> IndexedElement(A, i, j)
    A[i, j]

    It is recommended that ``IndexedElement`` objects be created via ``IndexedVariable``:

    >>> from pyccel.ast.core import IndexedElement
    >>> A = IndexedVariable('A')
    >>> IndexedElement(A, i, j) == A[i, j]
    False

    **todo:** fix bug. the last result must be : True
    """

    def __new__(
        cls,
        base,
        *args,
        **kw_args
        ):

        if not args:
            raise IndexException('Indexed needs at least one index.')
        if isinstance(base, (str, Symbol)):
            base = IndexedBase(base)
        elif not hasattr(base, '__getitem__') and not isinstance(base,
                IndexedBase):
            raise TypeError(filldedent("""
                Indexed expects string, Symbol, or IndexedBase as base."""))

        if isinstance(base, (NDimArray, Iterable, Tuple,
                      MatrixBase)) and all([i.is_number for i in args]):
            if len(args) == 1:
                return base[args[0]]
            else:
                return base[args]
        return Expr.__new__(cls, base, *args, **kw_args)

    def __init__(
        self,
        base,
        *args,
        **kw_args
        ):

        self._label = self._args[0]
        self._indices = self._args[1:]
        dtype = self.base.dtype
        shape = self.base.shape
        rank = self.base.rank
        self._precision = self.base.precision
        if isinstance(dtype, NativeInteger):
            self._dtype = NativeInteger()
        elif isinstance(dtype, NativeReal):
            self._dtype = NativeReal()
        elif isinstance(dtype, NativeComplex):
            self._dtype = NativeComplex()
        elif isinstance(dtype, NativeBool):
            self._dtype = NativeBool()
        elif isinstance(dtype, NativeString):
            self._dtype = NativeString()
        elif not isinstance(dtype, NativeRange):
            raise TypeError('Undefined datatype')

        if shape is not None:
            if self.order == 'C':
                shape = shape[::-1]
            new_shape = []
            for a,s in zip(args, shape):
                if isinstance(a, Slice):
                    start = a.start
                    end   = a.end
                    end   = s if end   is None else end
                    if start is None:
                        new_shape.append(end)
                    else:
                        new_shape.append(PyccelMinus(end, start))
            self._shape = tuple(new_shape)
            self._rank  = len(new_shape)
        else:
            new_rank = rank
            for i in range(rank):
                if not isinstance(args[i], Slice):
                    new_rank -= 1
            self._rank = new_rank

    @property
    def order(self):
        return self.base.order

    @property
    def base(self):
        return self._label

    @property
    def indices(self):
        return self._indices

class String(Basic, PyccelAstNode):

    """Represents the String"""
    _rank      = 0
    _shape     = ()
    _dtype     = NativeString()
    _precision = 0
    def __new__(cls, arg):
        if not isinstance(arg, str):
            raise TypeError('arg must be of type str')
        return Basic.__new__(cls, arg)

    @property
    def arg(self):
        return self._args[0]

    def __str__(self):
        return self.arg


class Concatenate(Basic, PyccelAstNode):

    """Represents the String concatination operation.

    Parameters
    ----------
    left : Symbol or string or List

    right : Symbol or string or List


    Examples
    --------
    >>> from sympy import symbols
    >>> from pyccel.ast.core import Concatenate
    >>> x = symbols('x')
    >>> Concatenate('some_string',x)
    some_string+x
    >>> Concatenate('some_string','another_string')
    'some_string' + 'another_string'
    """

    # TODO add step

    def __new__(cls, args, is_list):
        args = list(args)

        args = [ repr(arg) if isinstance(arg, str) else arg for arg in args]

        return Basic.__new__(cls, args, is_list)

    @property
    def args(self):
        return self._args[0]

    @property
    def is_list(self):
        return self._args[1]

    def _sympystr(self, printer):
        sstr = printer.doprint

        args = '+'.join(sstr(arg) for arg in self.args)

        return args




class Slice(Basic):

    """Represents a slice in the code.

    Parameters
    ----------
    start : Symbol or int
        starting index

    end : Symbol or int
        ending index

    Examples
    --------
    >>> from sympy import symbols
    >>> from pyccel.ast.core import Slice
    >>> m, n = symbols('m, n', integer=True)
    >>> Slice(m,n)
    m : n
    >>> Slice(None,n)
     : n
    >>> Slice(m,None)
    m :
    """

    # TODO add step
    # TODO check that args are integers
    # TODO add negative indices
    def __new__(cls, start, end):
        return Basic.__new__(cls, start, end)

    @property
    def start(self):
        return self._args[0]

    @property
    def end(self):
        return self._args[1]

    def _sympystr(self, printer):
        sstr = printer.doprint
        if self.start is None:
            start = ''
        else:
            start = sstr(self.start)
        if self.end is None:
            end = ''
        else:
            end = sstr(self.end)
        return '{0} : {1}'.format(start, end)

    def __str__(self):
        if self.start is None:
            start = ''
        else:
            start = str(self.start)
        if self.end is None:
            end = ''
        else:
            end = str(self.end)
        return '{0} : {1}'.format(start, end)

class Assert(Basic):

    """Represents a assert statement in the code.

    Parameters
    ----------
    test: Expr
        boolean expression to check

    Examples
    --------
    """
    #TODO add type check in the semantic stage
    def __new__(cls, test):
        #if not isinstance(test, (bool, Relational, sp_Boolean)):
        #    raise TypeError('test %s is of type %s, but must be a Relational, Boolean, or a built-in bool.'
        #                     % (test, type(test)))

        return Basic.__new__(cls, test)

    @property
    def test(self):
        return self._args[0]


class Eval(Basic):

    """Basic class for eval instruction."""

    pass


class Pass(Basic):

    """Basic class for pass instruction."""

    pass


class Exit(Basic):

    """Basic class for exists."""

    pass


class ErrorExit(Exit):

    """Exist with error."""

    pass


class If(Basic):

    """Represents a if statement in the code.

    Parameters
    ----------
    args :
        every argument is a tuple and
        is defined as (cond, expr) where expr is a valid ast element
        and cond is a boolean test.

    Examples
    --------
    >>> from sympy import Symbol
    >>> from pyccel.ast.core import Assign, If
    >>> n = Symbol('n')
    >>> If(((n>1), [Assign(n,n-1)]), (True, [Assign(n,n+1)]))
    If(((n>1), [Assign(n,n-1)]), (True, [Assign(n,n+1)]))
    """

    # TODO add type check in the semantic stage

    def __new__(cls, *args):

        newargs = []
        for ce in args:
            cond = ce[0]
            if PyccelAstNode.stage == 'semantic' and cond.dtype is not NativeBool():
                cond = PythonBool(cond)
            if isinstance(ce[1], (list, Tuple, tuple)):
                body = CodeBlock(ce[1])
            elif isinstance(ce[1], CodeBlock):
                body = ce[1]
            else:
                raise TypeError('body is not iterable or CodeBlock')
            newargs.append((cond,body))

        return Basic.__new__(cls, *newargs)

    @property
    def bodies(self):
        b = []
        for i in self._args:
            b.append( i[1])
        return b


class IfTernaryOperator(If):

    """class for the Ternery operator"""
    def __init__(self, *args):
        for arg in self.args:
            if len(arg[1].body)!=1:
                raise TypeError('IfTernary body must be of length 1')

    pass

class StarredArguments(Basic):
    def __new__(cls, args):
        return Basic.__new__(cls, args)

    @property
    def args_var(self):
        return self._args[0]


def is_simple_assign(expr):
    if not isinstance(expr, Assign):
        return False

    assignable = [Variable, IndexedVariable, IndexedElement]
    assignable += [sp_Integer, sp_Float]
    assignable = tuple(assignable)
    if isinstance(expr.rhs, assignable):
        return True
    else:
        return False


# ...

# ...

def get_initial_value(expr, var):
    """Returns the first assigned value to var in the Expression expr.

    Parameters
    ----------
    expr: Expression
        any AST valid expression

    var: str, Variable, DottedName, list, tuple
        variable name
    """

    # ...

    def is_None(expr):
        """Returns True if expr is None or Nil()."""

        return isinstance(expr, Nil) or expr is None

    # ...

    # ...

    if isinstance(var, str):
        return get_initial_value(expr, [var])
    elif isinstance(var, DottedName):

        return get_initial_value(expr, [str(var)])
    elif isinstance(var, Variable):

        return get_initial_value(expr, [var.name])
    elif not isinstance(var, (list, tuple)):

        raise TypeError('Expecting var to be str, list, tuple or Variable, given {0}'.format(type(var)))

    # ...

    # ...

    if isinstance(expr, ValuedVariable):
        if expr.variable.name in var:
            return expr.value
    elif isinstance(expr, Variable):

        # expr.cls_base if of type ClassDef

        if expr.cls_base:
            return get_initial_value(expr.cls_base, var)
    elif isinstance(expr, Assign):

        if str(expr.lhs) in var:
            return expr.rhs
    elif isinstance(expr, FunctionDef):

        value = get_initial_value(expr.body, var)
        if not is_None(value):
            r = get_initial_value(expr.arguments, value)
            if 'self._linear' in var:
                print ('>>>> ', var, value, r)
            if not r is None:
                return r
        return value

    elif isinstance(expr, ConstructorCall):

        return get_initial_value(expr.func, var)
    elif isinstance(expr, (list, tuple, Tuple)):

        for i in expr:
            value = get_initial_value(i, var)

            # here we make a difference between None and Nil,
            # since the output of our function can be None

            if not value is None:
                return value
    elif isinstance(expr, ClassDef):

        methods = expr.methods_as_dict
        init_method = methods['__init__']
        return get_initial_value(init_method, var)

    # ...

    return Nil()


# ...

# ... TODO treat other statements

def get_assigned_symbols(expr):
    """Returns all assigned symbols (as sympy Symbol) in the AST.

    Parameters
    ----------
    expr: Expression
        any AST valid expression
    """

    if isinstance(expr, (CodeBlock, FunctionDef, For, While)):
        return get_assigned_symbols(expr.body)
    elif isinstance(expr, FunctionalFor):
        return get_assigned_symbols(expr.loops)
    elif isinstance(expr, If):

        return get_assigned_symbols(expr.bodies)

    elif iterable(expr):
        symbols = []

        for a in expr:
            symbols += get_assigned_symbols(a)
        symbols = set(symbols)
        symbols = list(symbols)
        return symbols
    elif isinstance(expr, (Assign, AugAssign)):


        if expr.lhs is None:
            raise TypeError('Found None lhs')

        var = expr.lhs
        symbols = []
        if isinstance(var, DottedVariable):
            var = expr.lhs.lhs
            while isinstance(var, DottedVariable):
                var = var.lhs
            symbols.append(var)
        elif isinstance(var, IndexedElement):
            var = var.base
            symbols.append(var)
        elif isinstance(var, Variable):
            if var.rank:
                symbols.append(var)
        return symbols
    elif isinstance(expr, FunctionCall):
        f = expr.funcdef
        symbols = []
        for func_arg, inout in zip(expr.arguments,f.arguments_inout):
            if inout:
                symbols.append(func_arg)
        return symbols

    return []


# ...

# ... TODO: improve and make it recursive

def get_iterable_ranges(it, var_name=None):
    """Returns ranges of an iterable object."""

    if isinstance(it, Variable):
        if it.cls_base is None:
            raise TypeError('iterable must be an iterable Variable object'
                            )

        # ...

        def _construct_arg_Range(name):
            if not isinstance(name, DottedName):
                raise TypeError('Expecting a DottedName, given  {0}'.format(type(name)))

            if not var_name:
                return DottedName(it.name.name[0], name.name[1])
            else:
                return DottedName(var_name, name.name[1])

        # ...

        cls_base = it.cls_base

        if isinstance(cls_base, Range):
            if not isinstance(it.name, DottedName):
                raise TypeError('Expecting a DottedName, given  {0}'.format(type(it.name)))

            args = []
            for i in [cls_base.start, cls_base.stop, cls_base.step]:
                if isinstance(i, (Variable, IndexedVariable)):
                    arg_name = _construct_arg_Range(i.name)
                    arg = i.clone(arg_name)
                elif isinstance(i, IndexedElement):
                    arg_name = _construct_arg_Range(i.base.name)
                    base = i.base.clone(arg_name)
                    indices = i.indices
                    arg = base[indices]
                else:
                    raise TypeError('Wrong type, given {0}'.format(type(i)))
                args += [arg]

            return [Range(*args)]
        elif isinstance(cls_base, Tensor):

            if not isinstance(it.name, DottedName):
                raise TypeError('Expecting a DottedName, given  {0}'.format(type(it.name)))

            # ...

            ranges = []
            for r in cls_base.ranges:
                ranges += get_iterable_ranges(r,
                        var_name=str(it.name.name[0]))

            # ...

            return ranges

    elif isinstance(it, ConstructorCall):
        cls_base = it.this.cls_base

        # arguments[0] is 'self'
        # TODO must be improved in syntax, so that a['value'] is a sympy object

        args = []
        kwargs = {}
        for a in it.arguments[1:]:
            if isinstance(a, dict):

                # we add '_' tp be conform with the private variables convention

                kwargs['{0}'.format(a['key'])] = a['value']
            else:
                args.append(a)

        # TODO improve

        params = args

#        for k,v in kwargs:
#            params.append(k)

    methods = cls_base.methods_as_dict
    init_method = methods['__init__']

    args = init_method.arguments[1:]
    args = [str(i) for i in args]

    # ...

    it_method = methods['__iter__']
    targets = []
    starts = []
    for stmt in it_method.body:
        if isinstance(stmt, Assign):
            targets.append(stmt.lhs)
            starts.append(stmt.lhs)

    names = []
    for i in starts:
        if isinstance(i, IndexedElement):
            names.append(str(i.base))
        else:
            names.append(str(i))
    names = list(set(names))

    inits = {}
    for stmt in init_method.body:
        if isinstance(stmt, Assign):
            if str(stmt.lhs) in names:
                expr = stmt.rhs
                for (a_old, a_new) in zip(args, params):
                    dtype = datatype(stmt.rhs.dtype)
                    v_old = Variable(dtype, a_old)
                    if isinstance(a_new, (IndexedVariable,
                                  IndexedElement, str, Variable)):
                        v_new = Variable(dtype, a_new)
                    else:
                        v_new = a_new
                    expr = subs(expr, v_old, v_new)
                    inits[str(stmt.lhs)] = expr

    _starts = []
    for i in starts:
        if isinstance(i, IndexedElement):
            _starts.append(i.base)
        else:
            _starts.append(i)
    starts = [inits[str(i)] for i in _starts]

    # ...

    def _find_stopping_criterium(stmts):
        for stmt in stmts:
            if isinstance(stmt, If):
                if not len(stmt.args) == 2:
                    raise ValueError('Wrong __next__ pattern')

                (ct, et) = stmt.args[0]
                (cf, ef) = stmt.args[1]

                for i in et:
                    if isinstance(i, Raise):
                        return cf

                for i in ef:
                    if isinstance(i, Raise):
                        return ct

                raise TypeError('Wrong type for __next__ pattern')

        return None

    # ...

    # ...

    def doit(expr, targets):
        if isinstance(expr, Relational):
            if str(expr.lhs) in targets and expr.rel_op in ['<', '<=']:
                return expr.rhs
            elif str(expr.rhs) in targets and expr.rel_op in ['>', '>='
                    ]:
                return expr.lhs
            else:
                return None
        elif isinstance(expr, And):
            return [doit(a, targets) for a in expr.args]
        else:
            raise TypeError('Expecting And logical expression.')

    # ...

    # ...

    next_method = methods['__next__']
    ends = []
    cond = _find_stopping_criterium(next_method.body)

    # TODO treate case of cond with 'and' operation
    # TODO we should avoid using str
    #      must change target from DottedName to Variable

    targets = [str(i) for i in targets]
    ends = doit(cond, targets)

    # TODO not use str

    if not isinstance(ends, (list, tuple)):
        ends = [ends]

    names = []
    for i in ends:
        if isinstance(i, IndexedElement):
            names.append(str(i.base))
        else:
            names.append(str(i))
    names = list(set(names))

    inits = {}
    for stmt in init_method.body:
        if isinstance(stmt, Assign):
            if str(stmt.lhs) in names:
                expr = stmt.rhs
                for (a_old, a_new) in zip(args, params):
                    dtype = datatype(stmt.rhs.dtype)
                    v_old = Variable(dtype, a_old)
                    if isinstance(a_new, (IndexedVariable,
                                  IndexedElement, str, Variable)):
                        v_new = Variable(dtype, a_new)
                    else:
                        v_new = a_new
                    expr = subs(expr, v_old, v_new)
                    inits[str(stmt.lhs)] = expr

    _ends = []
    for i in ends:
        if isinstance(i, IndexedElement):
            _ends.append(i.base)
        else:
            _ends.append(i)
    ends = [inits[str(i)] for i in _ends]


    if not len(ends) == len(starts):
        raise ValueError('wrong number of starts/ends')

    # ...

    return [Range(s, e, 1) for (s, e) in zip(starts, ends)]

class ParserResult(Basic):
    def __new__(
        self,
        program=None,
        module=None,
        mod_name = None,
        prog_name = None,
        ):
        return Basic.__new__(self)

    def __init__(
        self,
        program=None,
        module=None,
        mod_name = None,
        prog_name = None,
        ):

        if program is not None  and not isinstance(program, CodeBlock):
            raise TypeError('Program must be a CodeBlock')

        if module is not None  and not isinstance(module, CodeBlock):
            raise TypeError('Module must be a CodeBlock')

        if program is not None and module is not None:
            if mod_name is None:
                raise TypeError('Please provide module name')
            elif not isinstance(mod_name, str):
                raise TypeError('Module name must be a string')
            if prog_name is None:
                raise TypeError('Please provide program name')
            elif not isinstance(prog_name, str):
                raise TypeError('Program name must be a string')

        self._program   = program
        self._module    = module
        self._prog_name = prog_name
        self._mod_name  = mod_name


    @property
    def program(self):
        return self._program

    @property
    def module(self):
        return self._module

    @property
    def prog_name(self):
        return self._prog_name

    @property
    def mod_name(self):
        return self._mod_name

    def has_additional_module(self):
        return self.program is not None and self.module is not None

    def is_program(self):
        return self.program is not None

    def get_focus(self):
        if self.is_program():
            return self.program
        else:
            return self.module


#==============================================================================
def process_shape(shape):
    if not hasattr(shape,'__iter__'):
        shape = [shape]

    new_shape = []
    for s in shape:
        if isinstance(s,(Py_Integer,Variable, Slice, PyccelAstNode, Function)):
            new_shape.append(s)
        elif isinstance(s, sp_Integer):
            new_shape.append(Py_Integer(s.p))
        elif isinstance(s, int):
            new_shape.append(Py_Integer(s))
        else:
            raise TypeError('shape elements cannot be '+str(type(s))+'. They must be one of the following types: Integer(pyccel), Variable, Slice, PyccelAstNode, Integer(sympy), int, Function')
    return tuple(new_shape)


class PyccelArraySize(Function, PyccelAstNode):
    def __new__(cls, arg, index):
        if not isinstance(arg, (list,
                                tuple,
                                Tuple,
                                PythonTuple,
                                List,
                                Variable,
                                IndexedElement,
                                IndexedBase)):
            raise TypeError('Uknown type of  %s.' % type(arg))

        return Basic.__new__(cls, arg, index)

    def __init__(self, arg, index):
        self._dtype = NativeInteger()
        self._rank  = 0
        self._shape = ()
        self._precision = default_precision['integer']

    @property
    def arg(self):
        return self._args[0]

    @property
    def index(self):
        return self._args[1]

    def _sympystr(self, printer):
        return 'Shape({},{})'.format(str(self.arg), str(self.index))

    def fprint(self, printer, lhs = None):
        """Fortran print."""

        lhs_code = printer(lhs)
        init_value = printer(self.arg)

        if self.arg.order == 'C':
            index = printer(self.arg.rank - self.index)
        else:
            index = printer(self.index + 1)

        if lhs:
            code_init = '{0} = size({1}, {2})'.format(lhs_code, init_value, index)
        else:
            code_init = 'size({0}, {1})'.format(init_value, index)

        return code_init
