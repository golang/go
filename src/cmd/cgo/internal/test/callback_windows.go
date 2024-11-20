// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

/*
#include <windows.h>
USHORT backtrace(ULONG FramesToCapture, PVOID *BackTrace) {
#ifdef _AMD64_
	CONTEXT context;
	RtlCaptureContext(&context);
	ULONG64 ControlPc;
	ControlPc = context.Rip;
	int i;
	for (i = 0; i < FramesToCapture; i++) {
		PRUNTIME_FUNCTION FunctionEntry;
		ULONG64 ImageBase;
		VOID *HandlerData;
		ULONG64 EstablisherFrame;

		FunctionEntry = RtlLookupFunctionEntry(ControlPc, &ImageBase, NULL);

		if (!FunctionEntry) {
			// For simplicity, don't unwind leaf entries, which are not used in this test.
			break;
		} else {
			RtlVirtualUnwind(0, ImageBase, ControlPc, FunctionEntry, &context, &HandlerData, &EstablisherFrame, NULL);
		}

		ControlPc = context.Rip;
		// Check if we left the user range.
		if (ControlPc < 0x10000) {
			break;
		}

		BackTrace[i] = (PVOID)(ControlPc);
	}
	return i;
#else
	return 0;
#endif
}
*/
import "C"

import (
	"internal/testenv"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"unsafe"
)

// Test that the stack can be unwound through a call out and call back
// into Go.
func testCallbackCallersSEH(t *testing.T) {
	testenv.SkipIfOptimizationOff(t) // This test requires inlining.
	if runtime.Compiler != "gc" {
		// The exact function names are not going to be the same.
		t.Skip("skipping for non-gc toolchain")
	}
	if runtime.GOARCH != "amd64" {
		// TODO: support SEH on other architectures.
		t.Skip("skipping on non-amd64")
	}
	// Only frames in the test package are checked.
	want := []string{
		"test._Cfunc_backtrace",
		"test.testCallbackCallersSEH.func1.1",
		// "test.testCallbackCallersSEH.func1", // hidden by inlining
		"test.goCallback",
		"test._Cfunc_callback",
		"test.nestedCall.func1",
		// "test.nestedCall", // hidden by inlining
		"test.testCallbackCallersSEH",
		"test.TestCallbackCallersSEH",
	}
	pc := make([]uintptr, 100)
	n := 0
	nestedCall(func() {
		n = int(C.backtrace(C.DWORD(len(pc)), (*C.PVOID)(unsafe.Pointer(&pc[0]))))
	})
	got := make([]string, 0, n)
	for i := 0; i < n; i++ {
		// This test is brittle in the face of inliner changes
		f := runtime.FuncForPC(pc[i] - 1)
		if f == nil {
			continue
		}
		fname := f.Name()
		switch fname {
		case "goCallback":
			// TODO(qmuntal): investigate why this function doesn't appear
			// when using the external linker.
			continue
		}
		// In module mode, this package has a fully-qualified import path.
		// Remove it if present.
		fname = strings.TrimPrefix(fname, "cmd/cgo/internal/")
		if !strings.HasPrefix(fname, "test.") {
			continue
		}
		got = append(got, fname)
	}
	if !reflect.DeepEqual(want, got) {
		t.Errorf("incorrect backtrace:\nwant:\t%v\ngot:\t%v", want, got)
	}
}
