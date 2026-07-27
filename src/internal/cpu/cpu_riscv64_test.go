// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build riscv64

package cpu_test

import (
	. "internal/cpu"
	"testing"
)

func TestRISCV64VectorLength(t *testing.T) {
	if RISCV64.HasV && RISCV64.VLENB == 0 {
		t.Fatal("VLENB should be non-zero when HasV is true")
	}
}
