// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ctfe

import (
	"cmd/compile/internal/syntax"
	"fmt"
	"testing"
)

// A view of an addressable array must follow writes to its existing storage.
// Copies of inline aggregates stay independent; reference fields keep sharing.
func TestAssignmentStorage(t *testing.T) {
	for _, tc := range []struct {
		name string
		body string
		want int64
	}{
		{"array_variable", `a := [1]int{1}; view := a[:]; a = [1]int{2}; return view[0]`, 2},
		{"parallel_array_variable", `a := [1]int{1}; view := a[:]; a, a[0] = [1]int{2}, 9; return view[0]*10 + a[0]`, 99},
		{"array_copy_independent", `a := [1]int{1}; b := a; av := a[:]; bv := b[:]; a = b; a[0] = 9; return av[0]*100 + bv[0]*10 + b[0]`, 911},
		{"nested_array_variable", `a := [1][2]int{{1, 2}}; view := a[0][:]; a = [1][2]int{{3, 4}}; return view[0]*10 + view[1]`, 34},
		{"array_element", `a := [1][1]int{{1}}; view := a[0][:]; a[0] = [1]int{2}; return view[0]`, 2},
		{"slice_element", `a := [][1]int{{1}}; view := a[0][:]; a[0] = [1]int{2}; return view[0]`, 2},
		{"struct_variable", `a := struct { A [1]int }{[1]int{1}}; view := a.A[:]; a = struct { A [1]int }{[1]int{2}}; return view[0]`, 2},
		{"struct_field", `a := struct { A [1]int }{[1]int{1}}; view := a.A[:]; a.A = [1]int{2}; return view[0]`, 2},
		{"array_struct_element", `a := [1]struct { A [1]int }{{[1]int{1}}}; view := a[0].A[:]; a[0] = struct { A [1]int }{[1]int{2}}; return view[0]`, 2},
		{"slice_struct_element", `a := []struct { A [1]int }{{[1]int{1}}}; view := a[0].A[:]; a[0] = struct { A [1]int }{[1]int{2}}; return view[0]`, 2},
		{"array_slice_references", `a := [1][]int{{1}}; b := a; view := a[:]; a = b; b[0][0] = 7; b[0] = []int{9}; return view[0][0]*100 + a[0][0]*10 + b[0][0]`, 779},
		{"array_map_references", `a := [1]map[int]int{{0: 1}}; b := a; view := a[:]; a = b; b[0][0] = 7; b[0] = map[int]int{0: 9}; return view[0][0]*100 + a[0][0]*10 + b[0][0]`, 779},
		{"range_assignment", `a := [1]int{0}; view := a[:]; for _, a = range [][1]int{{1}, {2}} {}; return view[0]`, 2},
		{"range_declaration_fresh", `views := [][]int{}; for _, a := range [][1]int{{1}, {2}} { views = append(views, a[:]) }; return views[0][0]*10 + views[1][0]`, 12},
		{"clear_views_and_independent_slots", `a := [][1]int{{1}, {2}}; first := a[0][:]; second := a[1][:]; clear(a); first[0] = 7; return a[0][0]*100 + second[0]*10 + a[1][0]`, 700},
		{"copy_overlapping_right", `a := [][1]int{{1}, {2}, {3}}; first := a[1][:]; second := a[2][:]; copy(a[1:], a[:2]); return first[0]*1000 + second[0]*100 + a[1][0]*10 + a[2][0]`, 1212},
		{"copy_overlapping_left", `a := [][1]int{{1}, {2}, {3}}; first := a[0][:]; second := a[1][:]; copy(a[:2], a[1:]); return first[0]*1000 + second[0]*100 + a[0][0]*10 + a[1][0]`, 2323},
		{"append_reused_capacity", `a := [][1]int{{1}, {2}}; view := a[1][:]; s := a[:1]; s = append(s, [1]int{3}); return view[0]*10 + s[1][0]`, 33},
		{"append_overlapping_capacity", `a := [][1]int{{1}, {2}, {3}, {4}}; first := a[1][:]; second := a[2][:]; s := a[:1]; s = append(s, a[:2]...); return first[0]*10 + second[0]`, 12},
		{"append_growth_copies_arrays", `a := [][1]int{{1}}; old := a[0][:]; a = append(a, [1]int{2}); next := a[0][:]; next[0] = 3; return old[0]*10 + next[0]`, 13},
		{"append_growth_initializes_capacity", `a := [][1]int{{1}, {2}}; a = append(a, [1]int{3}); a = a[:4]; view := a[3][:]; a[3] = [1]int{4}; return view[0]`, 4},
		{"zero_length_nested_array", `a := [1][0]int{{}}; view := a[:]; a = [1][0]int{{}}; return len(view)*10 + len(view[0])`, 10},
		{"struct_with_zero_length_array", `a := struct { Z [0]int; A [1]int }{[0]int{}, [1]int{1}}; view := a.A[:]; a = struct { Z [0]int; A [1]int }{[0]int{}, [1]int{2}}; return len(a.Z) + view[0]`, 2},
		{"map_array_value_independent", `m := map[int][1]int{0: {1}}; a := m[0]; view := a[:]; m[0] = [1]int{2}; a = [1]int{3}; return m[0][0]*100 + view[0]*10 + a[0]`, 233},
	} {
		for _, word := range []int64{4, 8} {
			t.Run(fmt.Sprintf("%s/word%d", tc.name, word), func(t *testing.T) {
				h := stmtHost(t, stmtIntFn("package p\nfunc Storage() int { "+tc.body+" }"))
				registerStorageStructs(t, h, h.funcs["Storage"].Body)
				v, _ := evalOne(t, h, Config{WordSize: word}, parseExpr(t, "Storage()"))
				mustInt(t, v, tc.want, "Storage()")
			})
		}
	}
}

// The fake host resolves built-in types already. These tests also use anonymous
// structs, including structs nested in array and slice element types.
func registerStorageStructs(t *testing.T, h *fakeHost, body *syntax.BlockStmt) {
	t.Helper()
	var typ func(syntax.Expr) *Type
	typ = func(x syntax.Expr) *Type {
		switch x := x.(type) {
		case *syntax.StructType:
			t := &Type{Kind: KStruct}
			for _, f := range x.FieldList {
				t.Fields = append(t.Fields, Field{Name: f.Name.Value, Type: typ(f.Type)})
			}
			return t
		case *syntax.ArrayType:
			n, err := parseIntLit(x.Len.(*syntax.BasicLit).Value, new(int64))
			if err != nil {
				t.Fatal(err)
			}
			return &Type{Kind: KArray, Len: n, Elem: typ(x.Elem)}
		case *syntax.SliceType:
			return &Type{Kind: KSlice, Elem: typ(x.Elem)}
		case *syntax.MapType:
			return &Type{Kind: KMap, Key: typ(x.Key), Elem: typ(x.Value)}
		}
		return h.resolveTypeExpr(x)
	}
	syntax.Inspect(body, func(n syntax.Node) bool {
		switch n := n.(type) {
		case *syntax.StructType:
			h.types[n] = typ(n)
		case *syntax.ArrayType:
			h.types[n] = typ(n)
		case *syntax.SliceType:
			h.types[n] = typ(n)
		}
		return true
	})
}
