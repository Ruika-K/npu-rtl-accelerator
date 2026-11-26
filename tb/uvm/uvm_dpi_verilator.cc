// The DPI part of the UVM library that this environment needs under Verilator:
// command-line arguments (+UVM_TESTNAME, ...) and regular expressions. The
// HDL-path backdoor (uvm_hdl.c) has no Verilator backend in the Accellera
// release and is not used here; the SystemVerilog side is compiled with
// +define+UVM_HDL_NO_DPI to match. Compile with -I<uvm-core>/src/dpi.
extern "C" {
#include <stdlib.h>
#include "uvm_dpi.h"

extern char* uvm_re_buffer();
extern const char* uvm_re_deglobbed(const char* glob, unsigned char with_brackets);
extern void uvm_re_free(regex_t* handle);
extern regex_t* uvm_re_comp(const char* re, unsigned char deglob);
extern int uvm_re_exec(regex_t* rexp, const char* str);
extern regex_t* uvm_re_compexec(const char* re, const char* str, unsigned char deglob, int* exec_ret);
const char* uvm_dpi_get_next_arg_c(int init);
extern char* uvm_dpi_get_tool_name_c();
extern char* uvm_dpi_get_tool_version_c();

#include "uvm_common.c"
#include "uvm_regex.cc"
#include "uvm_svcmd_dpi.c"
}
