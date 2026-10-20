// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"runtime"
	"testing"
	"unsafe"
)

// M20-S1 attack: the P carries fast-call pin slots and a call-start
// timestamp, and the M carries the in-fast-call flag. The enter helper
// records all three; the exit helper clears them. Red before the slice
// lands: the fields and the helpers do not exist.
func TestCgoFastcallPinSlots(t *testing.T) {
	var probe int
	arg := unsafe.Pointer(&probe)
	slots, stamp, flag, ok := runtime.LogosEnterFastCallForTest(arg, 12345)
	if !ok {
		t.Skip("no P in test context")
	}
	if !flag {
		t.Fatal("inFastCall not set by enter")
	}
	if want := [4]uintptr{uintptr(arg), 0, 0, 0}; slots != want {
		t.Fatalf("slots = %v, want %v", slots, want)
	}
	if stamp != 12345 {
		t.Fatalf("fastCallEnterNano = %d, want 12345", stamp)
	}

	slots, stamp, flag, ok = runtime.LogosExitFastCallForTest()
	if !ok {
		t.Skip("no P in test context")
	}
	if flag {
		t.Fatal("inFastCall still set after exit")
	}
	if slots != [4]uintptr{} {
		t.Fatalf("slots = %v after exit, want zeros", slots)
	}
	if stamp != 0 {
		t.Fatalf("fastCallEnterNano = %d after exit, want 0", stamp)
	}
}
