// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && (386 || arm || mips || mipsle)

package runtime_test

import (
	. "runtime"
	"testing"
	"unsafe"
)

func TestTimespec32Conversion(t *testing.T) {
	// The legacy futex and timer_settime syscalls still exchange two
	// signed 32-bit fields, including the full sub-second timeout.
	if got := unsafe.Sizeof(Timespec32{}); got != 8 {
		t.Fatalf("timespec32 size = %d, want 8", got)
	}
	for _, ns := range []int64{0, 1, 999999999, 1000000000, 1000000001, 2147483647999999999} {
		var ts Timespec
		ts.SetNsec(ns)
		ts32 := ts.DowncastToTimespec32()
		got := *(*[2]int32)(unsafe.Pointer(&ts32))
		want := [2]int32{int32(ns / 1000000000), int32(ns % 1000000000)}
		if got != want {
			t.Errorf("timespec32 conversion of %d: syscall representation = %v, want %v", ns, got, want)
		}
	}
}
