// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"internal/goexperiment"
	"internal/testenv"
	"runtime"
	"testing"
	"unsafe"
)

func TestCgoFastcallPinSlots(t *testing.T) {
	var probe int
	arg := unsafe.Pointer(&probe)
	slots, flag := runtime.LogosEnterFastCallForTest(arg)
	if !flag {
		t.Fatal("inFastCall not set by enter")
	}
	if want := [4]uintptr{uintptr(arg), 0, 0, 0}; slots != want {
		t.Fatalf("slots = %v, want %v", slots, want)
	}

	slots, flag = runtime.LogosExitFastCallForTest()
	if flag {
		t.Fatal("inFastCall still set after exit")
	}
	if slots != [4]uintptr{} {
		t.Fatalf("slots = %v after exit, want zeros", slots)
	}
}
func TestLogosFastCallRecordZeroesIdleSlots(t *testing.T) {
	runtime.LogosDirtyFastCallSlotsForTest([4]uintptr{11, 22, 33, 44}, false)
	var probe int
	arg := unsafe.Pointer(&probe)
	slots, _ := runtime.LogosEnterFastCallForTest(arg)
	defer runtime.LogosExitFastCallForTest()
	if want := [4]uintptr{uintptr(arg), 0, 0, 0}; slots != want {
		t.Fatalf("slots = %v, want %v", slots, want)
	}
}

func TestLogosFastCallClearsAllSlots(t *testing.T) {
	runtime.LogosDirtyFastCallSlotsForTest([4]uintptr{11, 22, 33, 44}, true)
	slots, flag := runtime.LogosExitFastCallForTest()
	if flag {
		t.Fatal("inFastCall still set after exit")
	}
	if slots != [4]uintptr{} {
		t.Fatalf("slots = %v after exit, want zeros", slots)
	}
}

func TestLogosFastCallPreservesCallerArgs(t *testing.T) {
	v := 12345
	arg := unsafe.Pointer(&v)
	runtime.LogosEnterFastCallForTest(arg)
	slots, _ := runtime.LogosFastCallStateForTest()
	if slots[0] != uintptr(arg) {
		runtime.LogosExitFastCallForTest()
		t.Fatalf("slot0 = %x, want %x", slots[0], uintptr(arg))
	}
	if v != 12345 {
		runtime.LogosExitFastCallForTest()
		t.Fatalf("caller value = %d, want 12345", v)
	}
	slots, flag := runtime.LogosExitFastCallForTest()
	if flag || slots != [4]uintptr{} {
		t.Fatalf("after exit = %v %v, want zeros", slots, flag)
	}
	if v != 12345 {
		t.Fatalf("caller value after exit = %d, want 12345", v)
	}
}

func TestLogosFastCallStaysOnOriginalM(t *testing.T) {
	var probe int
	arg := unsafe.Pointer(&probe)
	touched, recordOK, clearOK := runtime.LogosFastCallPassedMCheckForTest(arg)
	if !recordOK {
		t.Fatal("record did not set flag and slot on passed m")
	}
	if touched {
		t.Fatal("record touched current m instead of passed m")
	}
	if !clearOK {
		t.Fatal("clear did not zero the passed m")
	}
}

func TestLogosFastCgoFrameSlotsAreNotRoots(t *testing.T) {
	testenv.MustHaveCGO(t)
	t.Logf("logosfastcgo = %v", goexperiment.LogosFastCgo)
	if !goexperiment.LogosFastCgo {
		t.Skip("fast-call slot property is vacuous with GOEXPERIMENT=logosfastcgo off")
	}
	got := runTestProg(t, "testprogcgo", "LogosFastCgoSlotsNotRoots")
	want := "OK\n"
	if got != want {
		t.Fatalf("expected %q, but got:\n%s", want, got)
	}
}
