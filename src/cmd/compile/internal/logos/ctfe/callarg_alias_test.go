// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ctfe

import (
	"cmd/compile/internal/syntax"
	"testing"
)

func TestCallArgGoSemantics(t *testing.T) {
	for _, tt := range []struct {
		name   string
		callee string
		caller string
		want   int64
	}{
		{
			name:   "sliceMapSlot",
			callee: `func Touch(s []map[int]int) { s[0] = map[int]int{0: 42} }`,
			caller: `func Caller() int { s := []map[int]int{{0: 1}}; Touch(s); return s[0][0] }`,
			want:   42,
		},
		{
			name:   "sliceIntSlot",
			callee: `func Touch(s []int) { s[0] = 99 }`,
			caller: `func Caller() int { s := []int{1}; Touch(s); return s[0] }`,
			want:   99,
		},
		{
			name:   "arrayInlineSlot",
			callee: `func Touch(a [1]int) { a[0] = 99 }`,
			caller: `func Caller() int { a := [1]int{1}; Touch(a); return a[0] }`,
			want:   1,
		},
		{
			name:   "structInlineSlot",
			callee: `func Touch(s struct { N int }) { s.N = 99 }`,
			caller: `func Caller() int { s := struct { N int }{N: 1}; Touch(s); return s.N }`,
			want:   1,
		},
		{
			name:   "mapReferent",
			callee: `func Touch(m map[int]int) { m[0] = 99 }`,
			caller: `func Caller() int { m := map[int]int{0: 1}; Touch(m); return m[0] }`,
			want:   99,
		},
		{
			name:   "structMapReferent",
			callee: `func Touch(s struct { M map[int]int }) { m := s.M; m[0] = 99 }`,
			caller: `func Caller() int { s := struct { M map[int]int }{M: map[int]int{0: 1}}; Touch(s); return s.M[0] }`,
			want:   99,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			h := newFakeHost(t)
			callee := parseFunc(t, "package p\n"+tt.callee, true, nil, nil)
			caller := parseFunc(t, "package p\n"+tt.caller, true, nil, []*Type{BasicType("int")})
			h.funcs[callee.Name] = callee
			h.funcs[caller.Name] = caller
			syntax.Inspect(caller.Body, func(n syntax.Node) bool {
				if lit, ok := n.(*syntax.CompositeLit); ok {
					if st, ok := lit.Type.(*syntax.StructType); ok {
						field := st.FieldList[0]
						ft := h.resolveTypeExpr(field.Type)
						typ := &Type{Kind: KStruct, Fields: []Field{{Name: field.Name.Value, Type: ft}}}
						h.types[st] = typ
						h.types[lit] = typ
					}
				}
				return true
			})
			v, _ := evalOne(t, h, Config{}, parseExpr(t, `Caller()`))
			mustInt(t, v, tt.want, "Caller()")
		})
	}
}

func TestCallArgHostConstantProtected(t *testing.T) {
	h := newFakeHost(t)
	bootstrapHost(t, h, "M", `map[int]int{0: 1}`)
	callee := parseFunc(t, `package p
func Touch(m map[int]int) int { m[0] = 99; return m[0] }`, true, nil, []*Type{BasicType("int")})
	caller := parseFunc(t, `package p
func Caller() int {
	if Touch(M) != 99 { return -1 }
	return M[0]
}`, true, nil, []*Type{BasicType("int")})
	h.funcs[callee.Name] = callee
	h.funcs[caller.Name] = caller
	v, _ := evalOne(t, h, Config{}, parseExpr(t, `Caller()`))
	mustInt(t, v, 1, "later constant read in Caller()")
	v, _ = evalOne(t, h, Config{}, parseExpr(t, `M[0]`))
	mustInt(t, v, 1, "host constant after call")
}
