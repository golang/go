// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ctfe

import (
	"fmt"
	"strings"
	"testing"
)

// These cases exercise the two assignment phases rather than just swapping
// names: evaluate every destination and source, then perform stores in order.
// The expected results also run as ordinary Go in the source-level fixture.
func TestAssignmentEvaluationOrder(t *testing.T) {
	for _, tc := range []struct {
		name string
		body string
		want int64
	}{
		{
			name: "left_calls_before_right_calls",
			body: `log := []int{0}; dst := []int{0}; dst[Mark(log, 1)] = Mark(log, 2); return log[0]`,
			want: 12,
		},
		{
			name: "compound_left_calls_before_right_calls",
			body: `log := []int{0}; dst := []int{0}; dst[Mark(log, 1)] += Mark(log, 2); return log[0]`,
			want: 12,
		},
		{
			name: "index_base_before_index_operand",
			body: `log := []int{0}; Base(log)[Mark(log, 2)] = Mark(log, 3); return log[0]`,
			want: 123,
		},
		{
			name: "map_destination_survives_rebinding",
			body: `m := map[int]int{0: 1}; next := map[int]int{0: 2}; old := m; m, m[0] = next, 9; return old[0]*10 + m[0]`,
			want: 92,
		},
		{
			name: "nested_map_destination",
			body: `maps := []map[int]int{{0: 1}}; maps[0][0] = 9; return maps[0][0]`,
			want: 9,
		},
		{
			name: "comma_ok_destination_before_stores",
			body: `i := 0; flags := []bool{false, false}; m := map[int]int{7: 1}; i, flags[i] = m[7]; if flags[0] && !flags[1] { return i + 10 }; return -1`,
			want: 11,
		},
		{
			name: "comma_ok_destination_calls_before_lookup",
			body: `log := []int{0}; flags := []bool{false}; m := map[int]int{0: 7}; _, flags[Mark(log, 1)] = m[Mark(log, 2)]; if !flags[0] { return -1 }; return log[0]`,
			want: 12,
		},
		{
			name: "nil_map_comma_ok_declaration",
			body: `var m map[int]int; value, present := m[7]; if present { return -1 }; return value`,
			want: 0,
		},
		{
			name: "nil_map_comma_ok_assignment",
			body: `var m map[int]int; value, present := 9, true; value, present = m[7]; if present { return -1 }; return value`,
			want: 0,
		},
		{
			name: "map_lookup_missing_key",
			body: `m := map[int]int{0: 5}; value, present := m[7]; if present { return -1 }; return value`,
			want: 0,
		},
		{
			name: "slice_destination_survives_rebinding",
			body: `s := []int{1}; next := []int{2}; old := s; s, s[0] = next, 9; return old[0]*10 + s[0]`,
			want: 92,
		},
		{
			name: "array_destination_tracks_variable_storage",
			body: `a := [1]int{1}; a, a[0] = [1]int{2}, 9; return a[0]`,
			want: 9,
		},
		{
			name: "repeated_name_stores_left_to_right",
			body: `n := 0; n, n = 1, 2; return n`,
			want: 2,
		},
		{
			name: "blank_assignment_keeps_calls",
			body: `log := []int{0}; _, _ = Mark(log, 1), Mark(log, 2); return log[0]`,
			want: 12,
		},
	} {
		for _, word := range []int64{4, 8} {
			t.Run(fmt.Sprintf("%s/word%d", tc.name, word), func(t *testing.T) {
				h := stmtHost(t,
					stmtIntFn("package p\nfunc Mark(log []int, digit int) int { log[0] = log[0]*10 + digit; return 0 }", stmtSliceT, stmtIntT),
					stmtFunc{src: "package p\nfunc Base(log []int) []int { log[0] = log[0]*10 + 1; return []int{0} }", params: []*Type{stmtSliceT}, results: []*Type{stmtSliceT}},
					stmtIntFn("package p\nfunc Assignment() int { "+tc.body+" }"),
				)
				v, _ := evalOne(t, h, Config{WordSize: word}, parseExpr(t, "Assignment()"))
				mustInt(t, v, tc.want, "Assignment()")
			})
		}
	}
}

func TestAssignmentNilMapReportsError(t *testing.T) {
	h := stmtHost(t, stmtIntFn("package p\nfunc Assignment() int { var m map[int]int; m[0] = 1; return 0 }"))
	err := evalErr(t, h, Config{}, parseExpr(t, "Assignment()"))
	if err.Code != CodeEval || !strings.Contains(err.Detail, "nil map") || !err.Pos.IsKnown() {
		t.Fatalf("error = %#v, want a positioned nil-map assignment error", err)
	}
}
