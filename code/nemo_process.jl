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

    # Define the solver model (Cbc here; you can replace with Gurobi, CPLEX, etc.)
    jumpmodel = Model(Cbc.Optimizer)

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

    println("------------------------------------------------")
    println("NEMO termination status: ", status)
    println("Finished at:              ", Dates.now())
    println("------------------------------------------------")
end

main()
