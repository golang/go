#include <stdlib.h>

// It's a dummy function only used for linking runtime/cgo package,
// since crosscall2 is used in gcc_libinit.c, while it's implemented in asm_ARCH.S.
// Also, the linked object is only used to generate the _cgo_import.go file, by the "cgo -dynpackage" command,
// so it's safe to use a dummy function.
void crosscall2(void (*fn)(void *), void *a, int c, size_t ctxt) {
};
