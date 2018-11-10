// Copyright 2016 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

/*
struct {
	float x;
	_Complex float y;
} cplxAlign = { 3.14, 2.17 };
*/
import "C"

import "testing"

func TestComplexAlign(t *testing.T) {
	if C.cplxAlign.x != 3.14 {
		t.Errorf("got %v, expected 3.14", C.cplxAlign.x)
	}
	if C.cplxAlign.y != 2.17 {
		t.Errorf("got %v, expected 2.17", C.cplxAlign.y)
	}
}
