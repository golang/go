// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ctfe

import (
	"fmt"
	"testing"
)

// Returned aggregates are values before the next call can mutate their source.
// A named result can also escape through a slice view before its bare return.
func TestAssignmentStorageReturnedValues(t *testing.T) {
	arrayT := &Type{Kind: KArray, Len: 1, Elem: stmtIntT}
	sliceArrayT := &Type{Kind: KSlice, Elem: arrayT}
	sliceSliceT := &Type{Kind: KSlice, Elem: stmtSliceT}
	structT := &Type{Kind: KStruct, Fields: []Field{{Name: "A", Type: arrayT}, {Name: "S", Type: stmtSliceT}}}
	sliceStructT := &Type{Kind: KSlice, Elem: structT}
	for _, tc := range []struct {
		name string
		body string
		want int64
	}{
		{"array_result_rhs", `s := [][1]int{{1}}; a, _ := ReadArray(s), ReplaceArray(s); return a[0]`, 1},
		{"array_result_argument", `s := [][1]int{{1}}; return First(ReadArray(s), ReplaceArray(s))`, 1},
		{"array_result_composite", `s := [][1]int{{1}}; a := [2][1]int{ReadArray(s), MutateArray(s)}; return a[0][0]*10 + a[1][0]`, 12},
		{"array_result_append", `s := [][1]int{{1}}; a := append([][1]int{}, ReadArray(s), MutateArray(s)); return a[0][0]*10 + a[1][0]`, 12},
		{"named_array_result", `out := [][]int{{}}; a := ReadNamed(out); SetView(out); return a[0]`, 1},
		{"struct_result_preserves_reference_fields", `shared := []int{7}; s := []struct { A [1]int; S []int }{{[1]int{1}, shared}}; a, _ := ReadStruct(s), ReplaceStruct(s); shared[0] = 9; return a.A[0]*10 + a.S[0]`, 19},
		{"named_struct_result", `out := [][]int{{}}; a := ReadNamedStruct(out); SetView(out); return a.A[0]`, 1},
	} {
		for _, word := range []int64{4, 8} {
			t.Run(fmt.Sprintf("%s/word%d", tc.name, word), func(t *testing.T) {
				h := stmtHost(t,
					stmtFunc{src: "package p\n" + `func ReadArray(s [][1]int) [1]int { return s[0] }`, params: []*Type{sliceArrayT}, results: []*Type{arrayT}},
					stmtFunc{src: "package p\n" + `func ReplaceArray(s [][1]int) int { s[0] = [1]int{2}; return 0 }`, params: []*Type{sliceArrayT}, results: []*Type{stmtIntT}},
					stmtFunc{src: "package p\n" + `func MutateArray(s [][1]int) [1]int { s[0] = [1]int{2}; return s[0] }`, params: []*Type{sliceArrayT}, results: []*Type{arrayT}},
					stmtFunc{src: "package p\n" + `func First(a [1]int, n int) int { return a[0] }`, params: []*Type{arrayT, stmtIntT}, results: []*Type{stmtIntT}},
					stmtFunc{src: "package p\n" + `func ReadNamed(out [][]int) (a [1]int) { a[0] = 1; out[0] = a[:]; return }`, params: []*Type{sliceSliceT}, results: []*Type{arrayT}},
					stmtFunc{src: "package p\n" + `func SetView(out [][]int) int { out[0][0] = 2; return 0 }`, params: []*Type{sliceSliceT}, results: []*Type{stmtIntT}},
					stmtFunc{src: "package p\n" + `func ReadStruct(s []struct { A [1]int; S []int }) struct { A [1]int; S []int } { return s[0] }`, params: []*Type{sliceStructT}, results: []*Type{structT}},
					stmtFunc{src: "package p\n" + `func ReplaceStruct(s []struct { A [1]int; S []int }) int { s[0] = struct { A [1]int; S []int }{[1]int{2}, []int{3}}; return 0 }`, params: []*Type{sliceStructT}, results: []*Type{stmtIntT}},
					stmtFunc{src: "package p\n" + `func ReadNamedStruct(out [][]int) (a struct { A [1]int; S []int }) { a.A = [1]int{1}; out[0] = a.A[:]; return }`, params: []*Type{sliceSliceT}, results: []*Type{structT}},
					stmtIntFn("package p\nfunc Returned() int { "+tc.body+" }"),
				)
				for _, fn := range h.funcs {
					registerStorageStructs(t, h, fn.Body)
				}
				v, _ := evalOne(t, h, Config{WordSize: word}, parseExpr(t, "Returned()"))
				mustInt(t, v, tc.want, "Returned()")
			})
		}
	}
}
