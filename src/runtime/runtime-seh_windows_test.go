// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"internal/abi"
	"internal/syscall/windows"
	"runtime"
	"testing"
)

func sehf1() int {
	return sehf1()
}

func sehf2() {}

func TestSehLookupFunctionEntry(t *testing.T) {
	if runtime.GOARCH != "amd64" {
		t.Skip("skipping amd64-only test")
	}
	// This test checks that Win32 is able to retrieve
	// function metadata stored in the .pdata section
	// by the Go linker.
	// Win32 unwinding will fail if this test fails,
	// as RtlUnwindEx uses RtlLookupFunctionEntry internally.
	// If that's the case, don't bother investigating further,
	// first fix the .pdata generation.
	sehf1pc := abi.FuncPCABIInternal(sehf1)
	var fnwithframe func()
	fnwithframe = func() {
		fnwithframe()
	}
	fnwithoutframe := func() {}
	tests := []struct {
		name     string
		pc       uintptr
		hasframe bool
	}{
		{"no frame func", abi.FuncPCABIInternal(sehf2), false},
		{"no func", sehf1pc - 1, false},
		{"func at entry", sehf1pc, true},
		{"func in prologue", sehf1pc + 1, true},
		{"anonymous func with frame", abi.FuncPCABIInternal(fnwithframe), true},
		{"anonymous func without frame", abi.FuncPCABIInternal(fnwithoutframe), false},
		{"pc at func body", runtime.NewContextStub().GetPC(), true},
	}
	for _, tt := range tests {
		var base uintptr
		fn := windows.RtlLookupFunctionEntry(tt.pc, &base, nil)
		if !tt.hasframe {
			if fn != 0 {
				t.Errorf("%s: unexpected frame", tt.name)
			}
			continue
		}
		if fn == 0 {
			t.Errorf("%s: missing frame", tt.name)
		}
	}
}
