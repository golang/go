// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package demangle

import (
	"fmt"
	"testing"
)

func TestASTToString(t *testing.T) {
	var tests = []struct {
		input     AST
		want      string
		formatted string
	}{
		{
			&Qualified{Scope: &Name{Name: "s"}, Name: &Name{Name: "C"}},
			"s::C",
			`Qualified:
  Scope: s
  Name: C`,
		},
		{
			&Typed{Name: &Name{Name: "v"}, Type: &BuiltinType{"int"}},
			"int v",
			`Typed:
  Name: v
  Type: BuiltinType: int`,
		},
	}

	for i, test := range tests {
		if got := ASTToString(test.input); got != test.want {
			t.Errorf("ASTToString of test %d == %s, want %s", i, test.input, test.want)
		}
		if got := fmt.Sprintf("%#v", test.input); got != test.formatted {
			t.Errorf("Formatted test %d == %s, want %s", i, got, test.formatted)
		}
	}
}
