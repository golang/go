// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package completion

import (
	"go/types"
	"testing"
)

func TestFormatZeroValue(t *testing.T) {
	tests := []struct {
		typ  types.Type
		want string
	}{
		{types.Typ[types.String], `""`},
		{types.Typ[types.Byte], "0"},
		{types.Typ[types.Invalid], ""},
		{types.Universe.Lookup("error").Type(), "nil"},
	}

	for _, test := range tests {
		if got := formatZeroValue(test.typ, nil); got != test.want {
			t.Errorf("formatZeroValue(%v) = %q, want %q", test.typ, got, test.want)
		}
	}
}
