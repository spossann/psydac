# -*- coding: UTF-8 -*-

from mpi4py import MPI
from sympy import pi, cos, sin
from sympy.utilities.lambdify import implemented_function
import pytest

from sympde.calculus import grad, dot
from sympde.calculus import laplace
from sympde.topology import ScalarFunctionSpace
from sympde.topology import element_of
from sympde.topology import NormalVector
from sympde.topology import Square
from sympde.topology import Union
from sympde.expr     import BilinearForm, LinearForm, integral
from sympde.expr     import Norm
from sympde.expr     import find, EssentialBC

from psydac.fem.basic          import FemField
from psydac.api.discretization import discretize

#==============================================================================
def run_poisson_2d_dir(solution, f, ncells, degree, comm=None):

    # ... abstract model
    domain = Square()

    V = ScalarFunctionSpace('V', domain)

    F = element_of(V, name='F')

    v = element_of(V, name='v')
    u = element_of(V, name='u')

    int_0 = lambda expr: integral(domain , expr)

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), int_0(expr))

    expr = f*v
    l = LinearForm(v, int_0(expr))

    error = F - solution
    l2norm = Norm(error, domain, kind='l2')
    h1norm = Norm(error, domain, kind='h1')

    bc = EssentialBC(u, 0, domain.boundary)
    equation = find(u, forall=v, lhs=a(u,v), rhs=l(v), bc=bc)
    # ...

    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, ncells=ncells, comm=comm)
    # ...

    # ... discrete spaces
    Vh = discretize(V, domain_h, degree=degree)
    # ...

    # ... dsicretize the equation using Dirichlet bc
    equation_h = discretize(equation, domain_h, [Vh, Vh])
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, domain_h, Vh)
    h1norm_h = discretize(h1norm, domain_h, Vh)
    # ...

    # ... solve the discrete equation
    x = equation_h.solve()
    # ...

    # ...
    phi = FemField( Vh, x )
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)
    # ...

    return l2_error, h1_error

#==============================================================================
def run_poisson_2d_dirneu(solution, f, boundary, ncells, degree, comm=None):

    assert( isinstance(boundary, (list, tuple)) )

    # ... abstract model
    domain = Square()

    V = ScalarFunctionSpace('V', domain)

    B_neumann = [domain.get_boundary(**kw) for kw in boundary]
    if len(B_neumann) == 1:
        B_neumann = B_neumann[0]

    else:
        B_neumann = Union(*B_neumann)

    x,y = domain.coordinates

    F = element_of(V, name='F')

    v = element_of(V, name='v')
    u = element_of(V, name='u')

    nn = NormalVector('nn')

    int_0 = lambda expr: integral(domain , expr)
    int_1 = lambda expr: integral(B_neumann , expr)

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), int_0(expr))

    expr = f*v
    l0 = LinearForm(v, int_0(expr))

    expr = v*dot(grad(solution), nn)
    l_B_neumann = LinearForm(v, int_1(expr))

    expr = l0(v) + l_B_neumann(v)
    l = LinearForm(v, expr)

    error = F-solution
    l2norm = Norm(error, domain, kind='l2')
    h1norm = Norm(error, domain, kind='h1')

    B_dirichlet = domain.boundary.complement(B_neumann)
    bc = EssentialBC(u, 0, B_dirichlet)

    equation = find(u, forall=v, lhs=a(u,v), rhs=l(v), bc=bc)
    # ...

    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, ncells=ncells, comm=comm)
    # ...

    # ... discrete spaces
    Vh = discretize(V, domain_h, degree=degree)
    # ...

    # ... dsicretize the equation using Dirichlet bc
    equation_h = discretize(equation, domain_h, [Vh, Vh])
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, domain_h, Vh)
    h1norm_h = discretize(h1norm, domain_h, Vh)
    # ...

    # ... solve the discrete equation
    x = equation_h.solve()
    # ...

    # ...
    phi = FemField( Vh, x )
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)
    # ...

    return l2_error, h1_error

#==============================================================================
def run_laplace_2d_neu(solution, f, ncells, degree, comm=None):

    # ... abstract model
    domain = Square()

    V = ScalarFunctionSpace('V', domain)

    B_neumann = domain.boundary

    x,y = domain.coordinates

    F = element_of(V, name='F')

    v = element_of(V, name='v')
    u = element_of(V, name='u')

    nn = NormalVector('nn')

    int_0 = lambda expr: integral(domain , expr)
    int_1 = lambda expr: integral(B_neumann , expr)

    expr = dot(grad(v), grad(u)) + v*u
    a = BilinearForm((v,u), int_0(expr))

    expr = f*v
    l0 = LinearForm(v, int_0(expr))

    expr = v*dot(grad(solution), nn)
    l_B_neumann = LinearForm(v, int_1(expr))

    expr = l0(v) + l_B_neumann(v)
    l = LinearForm(v, expr)

    error = F-solution
    l2norm = Norm(error, domain, kind='l2')
    h1norm = Norm(error, domain, kind='h1')

    equation = find(u, forall=v, lhs=a(u,v), rhs=l(v))
    # ...

    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, ncells=ncells, comm=comm)
    # ...

    # ... discrete spaces
    Vh = discretize(V, domain_h, degree=degree)
    # ...

    # ... dsicretize the equation using Dirichlet bc
    equation_h = discretize(equation, domain_h, [Vh, Vh])
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, domain_h, Vh)
    h1norm_h = discretize(h1norm, domain_h, Vh)
    # ...

    # ... solve the discrete equation
    x = equation_h.solve()
    # ...

    # ...
    phi = FemField( Vh, x )
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)
    # ...

    return l2_error, h1_error

#==============================================================================
def run_biharmonic_2d_dir(solution, f, ncells, degree, comm=None):

    # ... abstract model
    domain = Square()

    V = ScalarFunctionSpace('V', domain)

    F = element_of(V, name='F')

    v = element_of(V, name='v')
    u = element_of(V, name='u')

    int_0 = lambda expr: integral(domain , expr)

    expr = laplace(v) * laplace(u)
    a = BilinearForm((v,u),int_0(expr))

    expr = f*v
    l = LinearForm(v, int_0(expr))

    error = F - solution
    l2norm = Norm(error, domain, kind='l2')
    h1norm = Norm(error, domain, kind='h1')

    nn = NormalVector('nn')
    bc  = [EssentialBC(u, 0, domain.boundary)]
    bc += [EssentialBC(dot(grad(u), nn), 0, domain.boundary)]
    equation = find(u, forall=v, lhs=a(u,v), rhs=l(v), bc=bc)
    # ...

    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, ncells=ncells, comm=comm)
    # ...

    # ... discrete spaces
    Vh = discretize(V, domain_h, degree=degree)
    # ...

    # ... dsicretize the equation using Dirichlet bc
    equation_h = discretize(equation, domain_h, [Vh, Vh])
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, domain_h, Vh)
    h1norm_h = discretize(h1norm, domain_h, Vh)
    # ...

    # ... solve the discrete equation
    x = equation_h.solve()
    # ...

    # ...
    phi = FemField( Vh, x )
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)
    # ...

    return l2_error, h1_error


#==============================================================================
def run_poisson_user_function_2d_dir(f, solution, ncells, degree, comm=None):

    # ... abstract model
    domain = Square()
    x,y = domain.coordinates

    f = implemented_function('f', f)

    V = ScalarFunctionSpace('V', domain)

    F = element_of(V, name='F')

    v = element_of(V, name='v')
    u = element_of(V, name='u')

    int_0 = lambda expr: integral(domain , expr)

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), int_0(expr))

    expr = f(x,y)*v
    l = LinearForm(v, int_0(expr))

    error = F - solution
    l2norm = Norm(error, domain, kind='l2')
    h1norm = Norm(error, domain, kind='h1')

    bc = EssentialBC(u, 0, domain.boundary)
    equation = find(u, forall=v, lhs=a(u,v), rhs=l(v), bc=bc)
    # ...

    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, ncells=ncells, comm=comm)
    # ...

    # ... discrete spaces
    Vh = discretize(V, domain_h, degree=degree)
    # ...

    # ... dsicretize the equation using Dirichlet bc
    equation_h = discretize(equation, domain_h, [Vh, Vh])
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, domain_h, Vh)
    h1norm_h = discretize(h1norm, domain_h, Vh)
    # ...

    # ... solve the discrete equation
    x = equation_h.solve()
    # ...

    # ...
    phi = FemField( Vh, x )
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)
    # ...

    return l2_error, h1_error


###############################################################################
#            SERIAL TESTS
###############################################################################

#==============================================================================
def test_api_poisson_2d_dir_1():

    from sympy.abc import x,y

    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*sin(pi*x)*sin(pi*y)

    l2_error, h1_error = run_poisson_2d_dir(solution, f,
                                            ncells=[2**3,2**3], degree=[2,2])

    expected_l2_error =  0.00021808678604760232
    expected_h1_error =  0.013023570720360362

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_poisson_2d_dirneu_1():

    from sympy.abc import x,y

    solution = cos(0.5*pi*x)*sin(pi*y)
    f        = (5./4.)*pi**2*solution

    l2_error, h1_error = run_poisson_2d_dirneu(solution, f, [{'axis': 0, 'ext': -1}],
                                               ncells=[2**3,2**3], degree=[2,2])

    expected_l2_error =  0.00015546057796452772
    expected_h1_error =  0.00926930278452745

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_poisson_2d_dirneu_2():

    from sympy.abc import x,y

    solution = sin(0.5*pi*x)*sin(pi*y)
    f        = (5./4.)*pi**2*solution

    l2_error, h1_error = run_poisson_2d_dirneu(solution, f, [{'axis': 0, 'ext': 1}],
                                               ncells=[2**3,2**3], degree=[2,2])

    expected_l2_error =  0.0001554605779481901
    expected_h1_error =  0.009269302784527256

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_poisson_2d_dirneu_3():

    from sympy.abc import x,y

    solution = sin(pi*x)*cos(0.5*pi*y)
    f        = (5./4.)*pi**2*solution

    l2_error, h1_error = run_poisson_2d_dirneu(solution, f, [{'axis': 1, 'ext': -1}],
                                               ncells=[2**3,2**3], degree=[2,2])

    expected_l2_error =  0.0001554605779681901
    expected_h1_error =  0.009269302784528678

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_poisson_2d_dirneu_4():

    from sympy.abc import x,y

    solution = sin(pi*x)*sin(0.5*pi*y)
    f        = (5./4.)*pi**2*solution

    l2_error, h1_error = run_poisson_2d_dirneu(solution, f, [{'axis': 1, 'ext': 1}],
                                               ncells=[2**3,2**3], degree=[2,2])

    expected_l2_error =  0.00015546057796339546
    expected_h1_error =  0.009269302784526841

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_poisson_2d_dirneu_13():

    from sympy.abc import x,y

    solution = cos(0.5*pi*x)*cos(0.5*pi*y)
    f        = (1./2.)*pi**2*solution

    l2_error, h1_error = run_poisson_2d_dirneu(solution, f,
                                               [{'axis': 0, 'ext': -1},
                                                {'axis': 1, 'ext': -1}],
                                               ncells=[2**3,2**3], degree=[2,2])

    expected_l2_error =  2.6119892736036942e-05
    expected_h1_error =  0.0016032430287934746

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_poisson_2d_dirneu_24():

    from sympy.abc import x,y

    solution = sin(0.5*pi*x)*sin(0.5*pi*y)
    f        = (1./2.)*pi**2*solution

    l2_error, h1_error = run_poisson_2d_dirneu(solution, f,
                                               [{'axis': 0, 'ext': 1},
                                                {'axis': 1, 'ext': 1}],
                                               ncells=[2**3,2**3], degree=[2,2])

    expected_l2_error =  2.611989253883369e-05
    expected_h1_error =  0.0016032430287973409

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_poisson_2d_dirneu_123():

    from sympy.abc import x,y

    solution = cos(pi*x)*cos(0.5*pi*y)
    f        = 5./4.*pi**2*solution

    l2_error, h1_error = run_poisson_2d_dirneu(solution, f,
                                               [{'axis': 0, 'ext': -1},
                                                {'axis': 0, 'ext': 1},
                                                {'axis': 1, 'ext': -1}],
                                               ncells=[2**3,2**3], degree=[2,2])

    expected_l2_error =  0.00015494478505412876
    expected_h1_error =  0.009242166414700994

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_poisson_2d_dir_zero_neu_nonzero_1():

    from sympy.abc import x,y

    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*solution

    l2_error, h1_error = run_poisson_2d_dirneu(solution, f, [{'axis': 0, 'ext': -1}],
                                               ncells=[2**3, 2**3], degree=[2, 2])

    expected_l2_error = 0.00021786960672322118
    expected_h1_error = 0.01302350067761091

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_poisson_2d_dir_zero_neu_nonzero_2():

    from sympy.abc import x,y

    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*solution

    l2_error, h1_error = run_poisson_2d_dirneu(solution, f, [{'axis': 0, 'ext': 1}],
                                               ncells=[2**3, 2**3], degree=[2, 2])

    expected_l2_error = 0.00021786960672322118
    expected_h1_error = 0.01302350067761091

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_poisson_2d_dir_zero_neu_nonzero_3():

    from sympy.abc import x,y

    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*solution

    l2_error, h1_error = run_poisson_2d_dirneu(solution, f, [{'axis': 1, 'ext': -1}],
                                               ncells=[2**3, 2**3], degree=[2, 2])

    expected_l2_error = 0.00021786960672322118
    expected_h1_error = 0.01302350067761091

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_poisson_2d_dir_zero_neu_nonzero_4():

    from sympy.abc import x,y

    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*solution

    l2_error, h1_error = run_poisson_2d_dirneu(solution, f, [{'axis': 1, 'ext': 1}],
                                               ncells=[2**3, 2**3], degree=[2, 2])

    expected_l2_error = 0.00021786960672322118
    expected_h1_error = 0.01302350067761091

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_laplace_2d_neu():

    from sympy.abc import x,y

    solution = cos(pi*x)*cos(pi*y)
    f        = (2.*pi**2 + 1.)*solution

    l2_error, h1_error = run_laplace_2d_neu(solution, f, ncells=[2**3,2**3], degree=[2,2])

    expected_l2_error =  0.0002172846538950129
    expected_h1_error =  0.012984852988125026

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_biharmonic_2d_dir_1():

    from sympy.abc import x,y
    from sympde.expr import TerminalExpr

    solution = (sin(pi*x)*sin(pi*y))**2

    # compute the analytical solution
    f = laplace(laplace(solution))
    f = TerminalExpr(f, dim=2)

    l2_error, h1_error = run_biharmonic_2d_dir(solution, f,
                                            ncells=[2**3,2**3], degree=[2,2])

    expected_l2_error =  0.015086415626061608
    expected_h1_error =  0.08773346232942228

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)



#==============================================================================
def test_api_poisson_user_function_2d_dir_1():

    from sympy.abc import x,y

    solution = sin(pi*x)*sin(pi*y)

    # ...
    def f(x,y):
        from numpy import pi
        from numpy import cos
        from numpy import sin

        value = 2*pi**2*sin(pi*x)*sin(pi*y)
        return value
    # ...

    l2_error, h1_error = run_poisson_user_function_2d_dir(f, solution,
                                            ncells=[2**3,2**3], degree=[2,2])

    expected_l2_error =  0.00021808678604760232
    expected_h1_error =  0.013023570720360362

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

###############################################################################
#            PARALLEL TESTS
###############################################################################

#==============================================================================
@pytest.mark.parallel
def test_api_poisson_2d_dir_1_parallel():

    from sympy.abc import x,y

    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*sin(pi*x)*sin(pi*y)

    l2_error, h1_error = run_poisson_2d_dir(solution, f,
                                            ncells=[2**3,2**3], degree=[2,2],
                                            comm=MPI.COMM_WORLD)

    expected_l2_error =  0.00021808678604760232
    expected_h1_error =  0.013023570720360362

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)


#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy import cache
    cache.clear_cache()

def teardown_function():
    from sympy import cache
    cache.clear_cache()
