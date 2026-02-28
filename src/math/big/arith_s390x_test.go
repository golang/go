// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !math_big_pure_go

package big

import "testing"

func TestAddVVNoVec(t *testing.T) {
	setDuringTest(t, &hasVX, false)
	TestAddVV(t)
}

func TestSubVVNoVec(t *testing.T) {
	setDuringTest(t, &hasVX, false)
	TestSubVV(t)
}
