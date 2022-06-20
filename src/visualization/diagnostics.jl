@userplot PlotDiagnostics
@recipe function f(diagnostics::PlotDiagnostics)
    @assert typeof(diagnostics.args[1]) <: FilterOutput
    standard_residuals = get_standard_residuals(diagnostics.args[1])
    @assert size(standard_residuals, 2) == 1
    n = length(standard_residuals)
    layout := (2, 2)
    acf = autocor(standard_residuals)[2:end]
    @series begin
        seriestype := :path
        label := ""
        seriescolor := "black"
        subplot := 1
        standard_residuals
    end
    @series begin
        seriestype := :bar
        label := ""
        seriescolor := "black"
        subplot := 2
        acf
    end
    ub = ones(length(acf)) * 1.96 / sqrt(n)
    lb = ones(length(acf)) * -1.96 / sqrt(n)
    @series begin
        seriestype := :path
        linestyle := :dash
        seriescolor := "red"
        label := ""
        subplot := 2
        ub
    end
    @series begin
        seriestype := :path
        linestyle := :dash
        label := ""
        seriescolor := "red"
        subplot := 2
        lb
    end
    @series begin
        seriestype := :histogram
        label := ""
        seriescolor := "black"
        normalize := true
        subplot := 3
        standard_residuals
    end
    normal(x) = pdf(Normal(), x)
    @series begin
        seriestype := :path
        label := ""
        seriescolor := "red"
        subplot := 3
        normal
    end
    qqpair = qqbuild(Normal(), standard_residuals)
    @series begin
        seriestype := :scatter
        label := ""
        seriescolor := "black"
        subplot := 4
        qqpair.qx, qqpair.qy
    end
    @series begin
        seriestype := :path
        seriescolor := "red"
        label := ""
        subplot := 4
        collect(-3:0.01:3), collect(-3:0.01:3)
    end
end