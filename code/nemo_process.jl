#!/usr/bin/env julia

# Simple wrapper script to run a NEMO scenario DB.
#
# Usage (from command line or Python subprocess):
#   julia nemo/run_nemo.jl path/to/scenario.sqlite
#
# Example:
#   julia nemo/run_nemo.jl data/usa_power_nemo.sqlite

using NemoMod
using JuMP
using Cbc
using Dates
import MathOptInterface as MOI

# Optional: use HiGHS for better infeasibility analysis if available
HIGHS_AVAILABLE = false
try
    @eval using HiGHS
    HIGHS_AVAILABLE = true
catch
    HIGHS_AVAILABLE = false
end

function build_jump_model()
    solver = get(ENV, "NEMO_SOLVER", "cbc") |> lowercase
    if solver == "highs" && HIGHS_AVAILABLE
        # Direct mode helps some solver tools
        return direct_model(HiGHS.Optimizer())
    end
    return Model(Cbc.Optimizer)
end

function main()
    if length(ARGS) < 1
        println("Usage: julia run_nemo.jl path/to/scenario.sqlite")
        exit(1)
    end

    dbpath = ARGS[1]

    println("------------------------------------------------")
    println("NEMO run started at: ", Dates.now())
    println("Scenario database:   ", dbpath)
    println("------------------------------------------------")

    # Define the solver model (default Cbc; set NEMO_SOLVER=highs to try HiGHS)
    jumpmodel = build_jump_model()
    # Silence solver iteration spam; comment out if you want full solver log.
    JuMP.set_silent(jumpmodel)

    # Call NEMO's main solve function
    status = NemoMod.calculatescenario(
        dbpath;
        jumpmodel = jumpmodel,
        # Optional tuning:
        # calcyears = [ [2017, 2018, 2019, 2020] ],
        # varstosave = "vdemandnn, vnewcapacity, vtotalcapacityannual,
        #               vproductionbytechnologyannual, vproductionnn,
        #               vusebytechnologyannual, vusenn, vtotaldiscountedcost",
        # restrictvars = true,
        # reportzeros = false,
    )

    # Optionally dump the model to LP for infeasibility/unbounded debugging.
    # Set env NEMO_WRITE_LP to a filepath to always write,
    # or leave unset to only write when status is not OPTIMAL.
    write_lp_path = get(ENV, "NEMO_WRITE_LP", "")
    should_write_lp = write_lp_path != "" || status != MOI.OPTIMAL
    if should_write_lp
        lp_path = write_lp_path != "" ? write_lp_path : "nemo_model_dump.lp"
        try
            JuMP.write_to_file(jumpmodel, lp_path)
            println("Wrote LP model to ", lp_path, " (status=", status, ")")
        catch err
            println("Failed to write LP model: ", err)
        end
    end

    if status != MOI.OPTIMAL
        println("Model status is ", status, " — running find_infeasibilities ...")
        try
            infeas = NemoMod.find_infeasibilities(jumpmodel, true)
            println("find_infeasibilities returned ", length(infeas), " constraint(s):")
            for c in infeas
                println("  ", c)
            end
        catch err
            println("find_infeasibilities failed: ", err)
        end
    end

    println("------------------------------------------------")
    println("NEMO termination status: ", status)
    println("Finished at:              ", Dates.now())
    println("------------------------------------------------")
end

main()
