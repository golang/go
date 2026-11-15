// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt

import (
	"testing"
	"unsafe"
)

func TestGroupLayout(t *testing.T) {
	if unsafe.Sizeof(uintptr(0)) != 8 {
		t.Skip("64-bit layout pins only")
	}
	if got := unsafe.Sizeof(Group{}); got != 152 {
		t.Fatalf("Sizeof(Group) = %d, want 152", got)
	}
	if got := unsafe.Sizeof(PanicError{}); got != 40 {
		t.Fatalf("Sizeof(PanicError) = %d, want 40", got)
	}
}
