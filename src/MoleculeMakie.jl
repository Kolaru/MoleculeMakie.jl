module MoleculeMakie

using AtomicSystems
using GeometryBasics
using LinearAlgebra
using Makie
using Mendeleev: Mendeleev, elements
using Unitful
using UnitfulAtomic

import PeriodicTable

export to_points
export plot_molecule!

function to_points(positions::AbstractVector)
    Point3f.(austrip.(positions))
end
to_points(positions::AbstractMatrix) = Point3f.(eachcol(austrip.(positions)))
to_points(positions::Observable) = @lift to_points($positions)

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

has_molecular_bond(atoms, pts::Observable, A, B ; tolerance = DEFAULT_BOND_TOLERANCE) =
    has_molecular_bond(atoms, pts[], A, B ; tolerance)


function has_molecular_bond(atoms, pts, A, B ; tolerance = DEFAULT_BOND_TOLERANCE)
    elemA = to_element(atoms[A])
    elemB = to_element(atoms[B])
    pA, pB = pts[A], pts[B]
    threshold = (1 + tolerance) * austrip(elemA.covalent_radius_pyykko + elemB.covalent_radius_pyykko) 
    return norm(pB - pA) <= threshold
end

function molecular_bonds(atoms, positions ; tolerance = DEFAULT_BOND_TOLERANCE)
    pts = to_points(positions)
    bonds = []
    for A in eachindex(atoms)
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
    pts = convert(Observable, pts)
    radius = convert(Float32, radius)

    mid = @lift ($pts[A] + $pts[B])/2
    cA = @lift normal_mesh(Cylinder($pts[A], $mid, radius))
    cB = @lift normal_mesh(Cylinder($mid, $pts[B], radius))
    return cA, cB
end

function plot_molecule!(ax, elems::Vector{Mendeleev.Element}, positions ;
        atom_size = 0.5f0,
        atom_radius = atom_size .* [E.atomic_radius_rahm / 154u"pm" for E in elems],
        bond_radius = atom_size / 4,
        alpha = 1,
        color = [(E.cpk_hex, alpha) for E in elems],
        bond_tolerance = DEFAULT_BOND_TOLERANCE,
        bonds = molecular_bonds(elems, positions ; tolerance = bond_tolerance),
        transparency = false,
        marker = :Sphere,
        kwargs...)

    positions = convert(Observable, positions)
    pts = @lift to_points($positions)

    meshscatter!(ax, pts ;
        color, 
        markersize = atom_radius,
        transparency,
        marker
    )
    
    for (A, B, visible) in bonds
        cylinderA, cylinderB = bond_cylinders(pts, A, B, bond_radius)
        mesh!(ax, cylinderA ; color = color[A], transparency, visible)
        mesh!(ax, cylinderB ; color = color[B], transparency, visible)
    end

    on(pts) do points
        for (A, B, visible) in bonds
            visible[] = has_molecular_bond(elems, points, A, B ; tolerance = bond_tolerance)
        end
    end
end

# Required to be able to use elements properties in default arguments
function plot_molecule!(ax, atoms, positions ; kwargs...)
    plot_molecule!(ax, to_element.(atoms), positions ; kwargs...)
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
