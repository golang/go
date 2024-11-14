// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fipstest

import (
	"crypto/internal/fips/alias"
	"testing"
)

var a, b [100]byte

var aliasingTests = []struct {
	x, y                       []byte
	anyOverlap, inexactOverlap bool
}{
	{a[:], b[:], false, false},
	{a[:], b[:0], false, false},
	{a[:], b[:50], false, false},
	{a[40:50], a[50:60], false, false},
	{a[40:50], a[60:70], false, false},
	{a[:51], a[50:], true, true},
	{a[:], a[:], true, false},
	{a[:50], a[:60], true, false},
	{a[:], nil, false, false},
	{nil, nil, false, false},
	{a[:], a[:0], false, false},
	{a[:10], a[:10:20], true, false},
	{a[:10], a[5:10:20], true, true},
}

func testAliasing(t *testing.T, i int, x, y []byte, anyOverlap, inexactOverlap bool) {
	any := alias.AnyOverlap(x, y)
	if any != anyOverlap {
		t.Errorf("%d: wrong AnyOverlap result, expected %v, got %v", i, anyOverlap, any)
	}
	inexact := alias.InexactOverlap(x, y)
	if inexact != inexactOverlap {
		t.Errorf("%d: wrong InexactOverlap result, expected %v, got %v", i, inexactOverlap, any)
	}
}

func TestAliasing(t *testing.T) {
	for i, tt := range aliasingTests {
		testAliasing(t, i, tt.x, tt.y, tt.anyOverlap, tt.inexactOverlap)
		testAliasing(t, i, tt.y, tt.x, tt.anyOverlap, tt.inexactOverlap)
	}
}
