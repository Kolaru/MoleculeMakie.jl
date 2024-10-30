# TODO This is currently not used as NormalModes has not been released
module MoleculeMakieNormalModesExt

using NormalModes

function animate_molecule(atoms, geometry, nm::NormalDecomposition ;
        bonds = molecular_bonds(atoms, geometry),
        period = 1.0,  # Period of the slowest mode
        nsigmas = 1,
        axkwargs = (;),
        kwargs...)

    fig = AnimatedFigure()
    ax = Axis3(fig[1, 1] ; aspect = :data, axkwargs...)

    periods = period * ustrip(frequencies(nm)[1] ./ frequencies(nm))
    modes = normal_modes(nm)
    σs = nsigmas * sqrt.(spatial_variances(nm)) / 2

    t = fig.time

    ws = @lift σs .* sin.(2π * $t ./ periods)
    u = @lift reshape(sum($ws' .* modes ; dims = 2), 3, :)
    pos = @lift $u + $geometry

    plot_molecule!(ax, atoms, pos ; bonds, kwargs...)

    fig
end

end