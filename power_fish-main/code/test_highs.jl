#!/usr/bin/env julia

using JuMP
using HiGHS

# Simple test to verify HiGHS works
model = direct_model(HiGHS.Optimizer())

@variable(model, x >= 0)
@variable(model, y >= 0)

@objective(model, Min, 2x + 3y)

@constraint(model, c1, x + y >= 1)
@constraint(model, c2, 2x + y >= 2)

optimize!(model)

println("Status: ", termination_status(model))
println("Optimal x: ", value(x))
println("Optimal y: ", value(y))
println("Optimal objective: ", objective_value(model))
