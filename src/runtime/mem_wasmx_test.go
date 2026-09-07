// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build wasm || wasm32

package runtime_test

import (
	"runtime"
	"testing"
)

func TestReserveAlignedSbrkOverflow(t *testing.T) {
	align := runtime.PhysPageSize
	for _, size := range []uintptr{^uintptr(0), ^uintptr(0) - align, ^uintptr(0) - align + 1} {
		p, n := runtime.SysReserveAlignedSbrk(size, align)
		if p != nil || n != 0 {
			t.Fatalf("reserve(%#x, %#x) = (%p, %#x), want (nil, 0)", size, align, p, n)
		}
	}
}
