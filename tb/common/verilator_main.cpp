// main() for every Verilator testbench (make unit-*, integ-*).
//
// Same loop as the one Verilator's --binary generates, plus one thing that
// loop does not do: with +cov=<file> it writes the coverage database (the
// cover properties of tb/sva) when the simulation ends. All testbenches are
// built with --prefix Vtop, so the model class has one name.
#include <memory>

#include "verilated.h"
#if VM_COVERAGE
#include "verilated_cov.h"
#endif
#include "Vtop.h"

int main(int argc, char** argv) {
    const std::unique_ptr<VerilatedContext> contextp{new VerilatedContext};
    contextp->commandArgs(argc, argv);
    const std::unique_ptr<Vtop> topp{new Vtop{contextp.get()}};

    while (!contextp->gotFinish()) {
        topp->eval();
        if (!topp->eventsPending()) break;
        contextp->time(topp->nextTimeSlot());
    }
    topp->final();

#if VM_COVERAGE
    // commandArgsPlusMatch returns the whole argument, "+cov=<file>", or ""
    const char* arg = contextp->commandArgsPlusMatch("cov=");
    if (arg[0] != '\0') contextp->coveragep()->write(arg + 5);
#endif
    return 0;
}
