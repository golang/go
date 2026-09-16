// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rand_test

import (
	"math"
	. "math/rand/v2"
	"testing"
)

func TestNewZipf(t *testing.T) {
	for _, test := range []struct {
		s, v    float64
		wantNil bool
	}{
		{math.NaN(), 1, true},
		{math.Inf(1), 1, true},
		{math.Inf(-1), 1, true},
		{2, math.NaN(), true},
		{2, math.Inf(1), true},
		{2, math.Inf(-1), true},
		{math.NaN(), math.NaN(), true},
		{math.Inf(1), math.Inf(1), true},
		{1, 1, true},
		{0, 1, true},
		{2, math.Nextafter(1, 0), true},
		{2, 0, true},
		{math.Nextafter(1, 2), 1, false},
		{2, 1, false},
		{2, 2, false},
	} {
		r := New(NewPCG(1, 2))
		if z := NewZipf(r, test.s, test.v, 10); (z == nil) != test.wantNil {
			t.Errorf("NewZipf(r, %v, %v, 10) == nil is %v, want %v", test.s, test.v, z == nil, test.wantNil)
		}
	}
}
