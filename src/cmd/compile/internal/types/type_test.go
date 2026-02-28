// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"testing"
)

func TestSSACompare(t *testing.T) {
	a := []*Type{
		TypeInvalid,
		TypeMem,
		TypeFlags,
		TypeVoid,
		TypeInt128,
	}
	for _, x := range a {
		for _, y := range a {
			c := x.Compare(y)
			if x == y && c != CMPeq || x != y && c == CMPeq {
				t.Errorf("%s compare %s == %d\n", x.extra, y.extra, c)
			}
		}
	}
}
