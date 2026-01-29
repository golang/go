// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"internal/abi"
	"internal/runtime/sys"
	"internal/syscall/windows"
	"runtime"
	"slices"
	"testing"
	"unsafe"
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
		{"pc at func body", sys.GetCallerPC(), true},
	}
	for _, tt := range tests {
		var base uintptr
		fn := windows.RtlLookupFunctionEntry(tt.pc, &base, nil)
		if !tt.hasframe {
			if fn != nil {
				t.Errorf("%s: unexpected frame", tt.name)
			}
			continue
		}
		if fn == nil {
			t.Errorf("%s: missing frame", tt.name)
		}
	}
}

//go:noinline
func newCtx() *windows.Context {
	var ctx windows.Context
	ctx.SetPC(sys.GetCallerPC())
	ctx.SetSP(sys.GetCallerSP())
	ctx.SetFP(runtime.GetCallerFp())
	return &ctx
}

func sehCallers() []uintptr {
	// We don't need a real context,
	// RtlVirtualUnwind just needs a context with
	// valid a pc, sp and fp (aka bp).
	ctx := newCtx()

	pcs := make([]uintptr, 15)
	var base, frame uintptr
	var n int
	for i := 0; i < len(pcs); i++ {
		fn := windows.RtlLookupFunctionEntry(ctx.PC(), &base, nil)
		if fn == nil {
			break
		}
		pcs[i] = ctx.PC()
		n++
		windows.RtlVirtualUnwind(0, base, ctx.PC(), fn, unsafe.Pointer(ctx), nil, &frame, nil)
	}
	return pcs[:n]
}

// SEH unwinding does not report inlined frames.
//
//go:noinline
func sehf3(pan bool) []uintptr {
	return sehf4(pan)
}

//go:noinline
func sehf4(pan bool) []uintptr {
	var pcs []uintptr
	if pan {
		panic("sehf4")
	}
	pcs = sehCallers()
	return pcs
}

func testSehCallersEqual(t *testing.T, pcs []uintptr, want []string) {
	t.Helper()
	got := make([]string, 0, len(want))
	for _, pc := range pcs {
		fn := runtime.FuncForPC(pc)
		if fn == nil || len(got) >= len(want) {
			break
		}
		name := fn.Name()
		switch name {
		case "runtime.panicmem":
			// These functions are skipped as they appear inconsistently depending
			// whether inlining is on or off.
			continue
		case "runtime_test.sehCallers":
			// This is an artifact of the implementation of sehCallers.
			continue
		}
		got = append(got, name)
	}
	if !slices.Equal(want, got) {
		t.Fatalf("wanted %v, got %v", want, got)
	}
}

func TestSehUnwind(t *testing.T) {
	if runtime.GOARCH != "amd64" {
		t.Skip("skipping amd64-only test")
	}
	pcs := sehf3(false)
	testSehCallersEqual(t, pcs, []string{"runtime_test.sehf4", "runtime_test.sehf3", "runtime_test.TestSehUnwind"})
}

func TestSehUnwindPanic(t *testing.T) {
	if runtime.GOARCH != "amd64" {
		t.Skip("skipping amd64-only test")
	}
	want := []string{"runtime_test.TestSehUnwindPanic.func1", "runtime.gopanic",
		"runtime_test.sehf4", "runtime_test.sehf3", "runtime_test.TestSehUnwindPanic"}
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("did not panic")
		}
		pcs := sehCallers()
		testSehCallersEqual(t, pcs, want)
	}()
	sehf3(true)
}

func TestSehUnwindDoublePanic(t *testing.T) {
	if runtime.GOARCH != "amd64" {
		t.Skip("skipping amd64-only test")
	}
	want := []string{"runtime_test.TestSehUnwindDoublePanic.func1.1", "runtime.gopanic",
		"runtime_test.TestSehUnwindDoublePanic.func1", "runtime.gopanic", "runtime_test.TestSehUnwindDoublePanic"}
	defer func() {
		defer func() {
			if recover() == nil {
				t.Fatal("did not panic")
			}
			pcs := sehCallers()
			testSehCallersEqual(t, pcs, want)
		}()
		if recover() == nil {
			t.Fatal("did not panic")
		}
		panic(2)
	}()
	panic(1)
}

func TestSehUnwindNilPointerPanic(t *testing.T) {
	if runtime.GOARCH != "amd64" {
		t.Skip("skipping amd64-only test")
	}
	want := []string{"runtime_test.TestSehUnwindNilPointerPanic.func1", "runtime.gopanic",
		"runtime.sigpanic", "runtime_test.TestSehUnwindNilPointerPanic"}
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("did not panic")
		}
		pcs := sehCallers()
		testSehCallersEqual(t, pcs, want)
	}()
	var p *int
	if *p == 3 {
		t.Fatal("did not see nil pointer panic")
	}
}
