module MoleculeMakie

using AtomicSystems
using GeometryBasics
using LinearAlgebra
using Makie
using Mendeleev: Mendeleev, elements
using Statistics, StatsBase
using Unitful
using UnitfulAtomic

import PeriodicTable

export to_points
export molecular_bonds
export plot_molecule!

function to_points(positions::AbstractVector)
    Point3f.(austrip.(positions))
end

to_points(positions::AbstractMatrix) = Point3f.(eachcol(austrip.(positions)))

function to_points(positions::Observable)
    pts = Observable.(to_points(positions[]))

    on(positions) do pos
        for (pt, new_pt) in zip(pts, to_points(pos))
            pt[] = new_pt
        end
    end
    return pts
end

"""
    to_element(element)

Convert a Symbol, Integer, String or PeriodicTable.Element to a
Mendeleev.Element.
"""
to_element(element) = element
to_element(A::Atom) = elements[A.element.number]
to_element(A::Union{Symbol, Integer, AbstractString}) = elements[A]
to_element(element::PeriodicTable.Element) = elements[element.number]

const DEFAULT_BOND_TOLERANCE = 0.0

function has_molecular_bond(atoms, pts, A, B ; tolerance = DEFAULT_BOND_TOLERANCE)
    elemA = to_element(atoms[A])
    elemB = to_element(atoms[B])
    pA, pB = pts[A], pts[B]
    threshold = (1 + tolerance) * austrip(elemA.covalent_radius_pyykko + elemB.covalent_radius_pyykko) 
    isa(pA, Observable) && return norm(pB[] - pA[]) <= threshold
    return norm(pB - pA) <= threshold
end

"""
    molecular_bonds(atoms, positions ; tolerance = DEFAULT_BOND_TOLERANCE)

Return a list of triplets of the form `(i, j, visible)`,
where `i` and `j` are indices of atoms and `visible` is an `Observable`
used to dynamically hide bonds.

The `visible` observable is updated when the input `positions` observable is updated.

A bond is present when the following condition is fullfilled:

`distance <= (1 + tolerance) * covalent_radius`

where `distance` is the distance between two atoms, `tolerance` is the input tolerance,
and `covalent_radius` is the sum of the `Mendeleev.Element.covalent_radius_pyykko`
for the two atoms considered.
"""
function molecular_bonds(atoms, positions ; tolerance = DEFAULT_BOND_TOLERANCE)
    pts = to_points(positions)
    bonds = []
    for A in 1:length(atoms)
        others = (A + 1):length(atoms)
        for B in others
            if has_molecular_bond(atoms, pts, A, B ; tolerance)
                push!(bonds, [A, B, Observable(true)])
            end
        end
    end
    return bonds
end

function bond_cylinders(pts, A, B, radius)
    radius = convert(Float32, radius)

    mid = @lift ($(pts[A]) + $(pts[B]))/2
    cA = @lift normal_mesh(Cylinder($(pts[A]), $mid, radius))
    cB = @lift normal_mesh(Cylinder($mid, $(pts[B]), radius))
    return cA, cB
end

function atom_inspector_label(atom)
    function label(plot, index, box)
        center = round.(mean(plot[1][].position), sigdigits = 3)
        return """$(atom.element.name) $(atom.name)
        Position: ($(center[1]), $(center[2]), $(center[3]))"""
    end
    return label
end

"""
    plot_molecule!(ax, system::AtomicSystem, positions ; kwargs...)

Plot a molecule in the given `Axis`.

The molecule is defined by the system and the position of the atoms,
where `positions is a `(3, n_atoms)` array.

Positions can be an `Observable`, the molecule and bonds will be updated accordingly.

Key word arguments
==================
- `atom_size` (default: `0.5`): scale of the representation.
  The default `atom_radius` and `bond_radius` are proportional to it.
  The default values are chosen to be reasonnable when the positions
  are given in atomic units.

- `atom_radius` (default: `atom_size * Mendeleev.Element.atomic_radius_rahm / 154u"pm"`):
  radius of the atoms in data coordinate.

- `bond_radius` (default: `atom_size / 4`):
  radius of the bond cylinder in data coordinate.

- `alpha` (default: 1): transparency of the plot.
 
- `color` (default: PeriodicTable.Element.cpk_hex):
  color of each atom.

- `bond_tolerance` (default: DEFAULT_BOND_TOLERANCE):
  the default relative tolerance given to `molecular_bonds`,
  which determines which atoms are linked by cylinders by default.

- `bonds` (default: `molecular_bonds`):
  list containing triplets of the form `(i, j, visible)`,
  where `i` and `j` are indices of atoms and `visible` is an `Observable`
  used to dynamically hide bonds.
  By default the function `molecular_bonds` is used to compute the bonds.
  The bonds produced in that way are updated when the positions of the atoms change,
  and can disappear if two atoms move away from each other.

- `transparency` (default: `false`):
  transparency argument passed to all `Makie` 3D plots. See e.g. `mesh` for detail.
"""
function plot_molecule!(ax, system::AtomicSystem, positions ;
        atom_size = 0.5f0,
        atom_radius = atom_size .* [E.atomic_radius_rahm / 154u"pm" for E in to_element.(system)],
        bond_radius = atom_size / 4,
        alpha = 1,
        color = [(A.element.cpk_hex, alpha) for A in system],
        bond_tolerance = DEFAULT_BOND_TOLERANCE,
        bonds = molecular_bonds(to_element.(system), positions ; tolerance = bond_tolerance),
        transparency = false,
        kwargs...)

    positions = convert(Observable, positions)
    pts = to_points(positions)

    for (pt, atom, rad, col) in zip(pts, system, atom_radius, color)
        mesh!(ax, @lift(Sphere($pt, rad)) ;
            color = col,
            transparency,
            inspector_label = atom_inspector_label(atom)
        )
    end
    
    for (A, B, visible) in bonds
        cylinderA, cylinderB = bond_cylinders(pts, A, B, bond_radius)
        mesh!(ax, cylinderA ;
            color = color[A],
            transparency, visible,
            inspector_hover = Returns(false))
        mesh!(ax, cylinderB ;
            color = color[B],
            transparency, visible,
            inspector_hover = Returns(false))
    end

    onany(pts...) do points...
        for (A, B, visible) in bonds
            visible[] = has_molecular_bond(to_element.(system), points, A, B ; tolerance = bond_tolerance)
        end
    end
end

function plot_molecule_mode!(
        ax, atoms, positions, mode, t ;
        amplitude = 1.0,  # Amplitude of the animation
        period = 2.0,  # Animation period in second
        amplifications = ones(length(atoms)),
        bonds = molecular_bonds(atoms, positions),
        reframe = identity,
        kwargs...)

    t = convert(Observable, t)

    amplifications = reshape(amplifications, 1, :)
    w = @lift $amplitude * sin(2π * $t/$period)
    u = @lift $w * $mode .* amplifications
    pos = @lift reframe($u + $positions .* amplifications)

    plot_molecule!(ax, atoms, pos ; bonds, kwargs...)
end

function trace_molecule_mode!(
        ax, atoms, positions, mode ;
        amplifications = ones(length(atoms)),
        amplitude = 2,
        trace_amplitudes = range(0, amplitude ; length = 5),
        bonds = molecular_bonds(atoms, positions),
        trace_alpha = 0.2,
        kwargs...)
    
    amp = reshape(amplifications, 1, :)

    pos = positions .* amp
    plot_molecule!(ax, atoms, pos ; bonds, kwargs...)

    for w in trace_amplitudes
        pos = (w * mode + positions) .* amp
        plot_molecule!(ax, atoms, pos ;
            bonds,
            alpha = trace_alpha,
            transparency = false,
            kwargs...
        )
    end
end

function animate_molecule_mode(atoms, positions, mode ; kwargs...)
    fig = AnimatedFigure()
    ax = Axis3(fig[1, 1] ; aspect = :data)
    plot_molecule_mode!(ax, atoms, positions, mode, fig.time ; kwargs...)

    return fig
end

function animate_molecule_modes(atoms, positions, modes ; kwargs...)
    fig = AnimatedFigure()
    n_modes = size(modes, 3)

    for k in 1:n_modes
        ax = Axis3(fig[1, k] ; aspect = :data)
        plot_molecule_mode!(ax, atoms, positions, modes[:, :, k], fig.time ; kwargs...)
    end
    fig
end

function animated_geometry(geometry, component, tick ;
        period = 1.0,
        σ = 1.0)

    w = @lift σ* sin(2π * $tick.time / period)
    return @lift geometry + component * $w
end

end # module MoleculeMakie
