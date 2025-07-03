// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unix_test

import (
	"internal/goarch"
	"internal/syscall/unix"
	"runtime"
	"strings"
	"testing"
	"unsafe"
)

// TestSiginfoChildLayout validates SiginfoChild layout. Modelled after
// static assertions in linux kernel's arch/*/kernel/signal*.c.
func TestSiginfoChildLayout(t *testing.T) {
	var si unix.SiginfoChild

	const host64bit = goarch.PtrSize == 8

	if v := unsafe.Sizeof(si); v != 128 {
		t.Fatalf("sizeof: got %d, want 128", v)
	}

	ofSigno := 0
	ofErrno := 4
	ofCode := 8
	if strings.HasPrefix(runtime.GOARCH, "mips") {
		// These two fields are swapped on MIPS platforms.
		ofErrno, ofCode = ofCode, ofErrno
	}
	ofPid := 12
	if host64bit {
		ofPid = 16
	}
	ofUid := ofPid + 4
	ofStatus := ofPid + 8

	offsets := []struct {
		name string
		got  uintptr
		want int
	}{
		{"Signo", unsafe.Offsetof(si.Signo), ofSigno},
		{"Errno", unsafe.Offsetof(si.Errno), ofErrno},
		{"Code", unsafe.Offsetof(si.Code), ofCode},
		{"Pid", unsafe.Offsetof(si.Pid), ofPid},
		{"Uid", unsafe.Offsetof(si.Uid), ofUid},
		{"Status", unsafe.Offsetof(si.Status), ofStatus},
	}

	for _, tc := range offsets {
		if int(tc.got) != tc.want {
			t.Errorf("offsetof %s: got %d, want %d", tc.name, tc.got, tc.want)
		}
	}
}
