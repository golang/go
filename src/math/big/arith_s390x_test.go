// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build s390x,!math_big_pure_go

package big

import (
	"testing"
)

// Tests whether the non vector routines are working, even when the tests are run on a
// vector-capable machine

func TestFunVVnovec(t *testing.T) {
	if hasVX == true {
		for _, a := range sumVV {
			arg := a
			testFunVV(t, "addVV_novec", addVV_novec, arg)

			arg = argVV{a.z, a.y, a.x, a.c}
			testFunVV(t, "addVV_novec symmetric", addVV_novec, arg)

			arg = argVV{a.x, a.z, a.y, a.c}
			testFunVV(t, "subVV_novec", subVV_novec, arg)

			arg = argVV{a.y, a.z, a.x, a.c}
			testFunVV(t, "subVV_novec symmetric", subVV_novec, arg)
		}
	}
}
