// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Disabled for s390x because it uses assembly routines that are not
// accurate for huge arguments.

//go:build !s390x
// +build !s390x

package cmplx

import (
	"testing"
)

func TestTanHuge(t *testing.T) {
	for i, x := range hugeIn {
		if f := Tan(x); !cSoclose(tanHuge[i], f, 3e-15) {
			t.Errorf("Tan(%g) = %g, want %g", x, f, tanHuge[i])
		}
	}
}
