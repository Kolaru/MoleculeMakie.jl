module MoleculeMakie

using GeometryBasics
using LinearAlgebra
using Makie
using Mendeleev: Mendeleev, elements
using Unitful
using UnitfulAtomic

import PeriodicTable

function to_points(positions::AbstractVector)
    Point3f.(positions)
end
to_points(positions::AbstractMatrix) = Point3f.(eachcol(positions))

"""
    to_element(element)

Convert a Symbol, Integer, String or PeriodicTable.Element to a
Mendeleev.Element.
"""
to_element(element) = element
to_element(A::Union{Symbol, Integer, AbstractString}) = elements[A]
to_element(element::PeriodicTable.Element) = elements[element.number]

function molecular_bonds(atoms, positions ; tolerance = 0.1)
    pts = to_points(positions)
    elems = to_element.(atoms)
    bonds = []
    for (A, elemA) in enumerate(elems)
        others = (A + 1):length(elems)
        for (B, elemB) in zip(others, elems[others])
            threshold = (1 + tolerance) * austrip(elemA.covalent_radius_pyykko + elemB.covalent_radius_pyykko) 
            pA, pB = pts[][A], pts[][B]

            if norm(pB - pA) <= threshold
                push!(bonds, [A, B])
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
        atom_radius = atom_size * [E.atomic_radius_rahm / 154u"pm" for E in elems],
        bond_radius = atom_size / 4,
        alpha = 1,
        color = [(E.cpk_hex, alpha) for E in elems],
        bonds = molecular_bonds(elems, positions),
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
    
    for (A, B) in bonds
        cylinderA, cylinderB = bond_cylinders(pts, A, B, bond_radius)
        mesh!(ax, cylinderA ; color = color[A], transparency)
        mesh!(ax, cylinderB ; color = color[B], transparency)
    end
end

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

    w = @lift σ * sin(2π * $tick.time / period)
    return @lift geometry + component * $w
end

end # module MoleculeMakie
