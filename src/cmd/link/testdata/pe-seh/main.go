// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

/*
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

// Use assembly to ensure .pdata refers to a named symbol within .xdata,
// rather than the section symbol plus an addend. This exercises the
// linker's interior-symbol handling independently of the C compiler's unwind layout.
__asm__(
	// Match the order of the manually emitted and compiler-generated .pdata.
	".text\n"
		".globl seh_test_function, seh_test_unwind\n"
		"seh_test_function: nop; ret\n"
	"seh_test_function_end:\n"

	// Keep the handler in its own section, referenced only by unwind data.
	".section .text$h,\"xr\"\n"
		"seh_test_handler: movl $1, %eax; ret\n" // ExceptionContinueSearch

	".section .xdata,\"dr\"\n"
		".long 0\n" // Give the interior symbol a nonzero offset.
		"seh_test_unwind: .long 9\n" // Version 1, UNW_FLAG_EHANDLER, no unwind codes.
	".rva seh_test_handler\n"

	".section .pdata,\"dr\"\n"
		".rva seh_test_function, seh_test_function_end, seh_test_unwind\n"
	".text\n"
);

extern void seh_test_function(void);

static const char *check(void) {
	CONTEXT context = {0};
	DWORD64 stack[1] = {0};
	DWORD64 imageBase, frame;
	PVOID handlerData;
	context.ContextFlags = CONTEXT_FULL;
	context.Rip = (DWORD64)seh_test_function;
	context.Rsp = (DWORD64)stack;
	PRUNTIME_FUNCTION function = RtlLookupFunctionEntry(context.Rip, &imageBase, NULL);
	if (function == NULL)
		return "missing runtime function entry";
	PEXCEPTION_ROUTINE handler = RtlVirtualUnwind(UNW_FLAG_EHANDLER,
		imageBase, context.Rip, function, &context, &handlerData, &frame, NULL);
	if (handler == NULL || (DWORD64)handler == imageBase)
		return "exception handler was discarded";
	if (handler(NULL, NULL, &context, NULL) != ExceptionContinueSearch)
		return "unexpected exception handler result";
	return NULL;
}
*/
import "C"

func main() {
	if err := C.check(); err != nil {
		panic(C.GoString(err))
	}
}
