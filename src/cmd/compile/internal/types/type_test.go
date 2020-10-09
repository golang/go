// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types_test

import (
	"cmd/compile/internal/types"
	"testing"
)

func TestSSACompare(t *testing.T) {
	a := []*types.Type{
		types.TypeInvalid,
		types.TypeMem,
		types.TypeFlags,
		types.TypeVoid,
		types.TypeInt128,
	}
	for _, x := range a {
		for _, y := range a {
			c := x.Compare(y)
			if x == y && c != types.CMPeq || x != y && c == types.CMPeq {
				t.Errorf("%s compare %s == %d\n", x.Extra, y.Extra, c)
			}
		}
	}
}
