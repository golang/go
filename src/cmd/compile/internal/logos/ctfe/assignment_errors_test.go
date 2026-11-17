// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ctfe

import (
	"strings"
	"testing"
)

func TestAssignmentFailureOrder(t *testing.T) {
	for _, tc := range []struct {
		name string
		body string
		want string
	}{
		{"slice_store_after_rhs", `s := []int{0}; s[2] = Boom("rhs"); return 0`, "rhs"},
		{"nil_slice_store_after_rhs", `var s []int; s[0] = Boom("rhs"); return 0`, "rhs"},
		{"nil_map_store_after_rhs", `var m map[int]int; m[0] = Boom("rhs"); return 0`, "rhs"},
		{"destination_call_before_rhs", `s := []int{0}; s[Boom("lhs")] = Boom("rhs"); return 0`, "lhs"},
		{"index_base_before_index_call", `Base()[Boom("index")] = Boom("rhs"); return 0`, "base"},
		{"slice_store_bounds", `s := []int{0}; s[1] = 2; return 0`, "index out of range"},
		{"nil_slice_store_bounds", `var s []int; s[0] = 2; return 0`, "index out of range"},
		{"array_store_bounds", `a := [1]int{0}; i := 1; a[i] = 2; return 0`, "index out of range"},
		{"nil_map_compound_store", `var m map[int]int; m[0] += 2; return 0`, "nil map"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := stmtHost(t,
				stmtIntFn("package p\nfunc Boom(s string) int { panic(s) }", BasicType("string")),
				stmtFunc{src: "package p\nfunc Base() []int { panic(\"base\") }", results: []*Type{stmtSliceT}},
				stmtIntFn("package p\nfunc Assignment() int { "+tc.body+" }"),
			)
			err := evalErr(t, h, Config{}, parseExpr(t, "Assignment()"))
			if err.Code != CodeEval || !strings.Contains(err.Detail, tc.want) || !err.Pos.IsKnown() {
				t.Fatalf("error = %#v, want positioned CodeEval containing %q", err, tc.want)
			}
		})
	}
}
