// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-license
// file in the LICENSE file.

package ctfe

import "testing"

func TestSpreadTypedNilSliceBindsZeroLen(t *testing.T) {
	intT := BasicType("int")
	sliceT := &Type{Kind: KSlice, Elem: intT}

	h := newFakeHost(t)
	fn := parseFunc(t, "package p\nfunc VLen(xs ...int) int { return len(xs) }",
		true, []*Type{sliceT}, []*Type{intT})
	fn.Variadic = true
	h.funcs["VLen"] = fn
	caller := parseFunc(t, "package p\nfunc CallNilLen() int { var s []int; return VLen(s...) }",
		true, nil, []*Type{intT})
	h.funcs[caller.Name] = caller

	v, _ := evalOne(t, h, Config{}, parseExpr(t, `CallNilLen()`))
	if got := v.Int64(); got != 0 {
		t.Fatalf("CallNilLen() = %d, want 0", got)
	}
}

func TestSpreadNilVsEmptyIdentity(t *testing.T) {
	intT := BasicType("int")
	sliceT := &Type{Kind: KSlice, Elem: intT}
	boolT := BasicType("bool")

	h := newFakeHost(t)
	fn := parseFunc(t, "package p\nfunc IsNil(xs ...int) bool { return xs == nil }",
		true, []*Type{sliceT}, []*Type{boolT})
	fn.Variadic = true
	h.funcs["IsNil"] = fn
	nilCaller := parseFunc(t, "package p\nfunc CallNilIsNil() bool { var s []int; return IsNil(s...) }",
		true, nil, []*Type{boolT})
	h.funcs[nilCaller.Name] = nilCaller
	emptyCaller := parseFunc(t, "package p\nfunc CallEmptyIsNil() bool { return IsNil([]int{}...) }",
		true, nil, []*Type{boolT})
	h.funcs[emptyCaller.Name] = emptyCaller

	cfg := Config{}
	v, _ := evalOne(t, h, cfg, parseExpr(t, `CallNilIsNil()`))
	if !v.Bool() {
		t.Fatalf("CallNilIsNil() = false, want true")
	}
	v, _ = evalOne(t, h, cfg, parseExpr(t, `CallEmptyIsNil()`))
	if v.Bool() {
		t.Fatalf("CallEmptyIsNil() = true, want false")
	}
}

func TestAppendUntypedStringSpread(t *testing.T) {
	intT := BasicType("int")

	h := newFakeHost(t)
	fn := parseFunc(t, "package p\nfunc AppendUntyped() int {\n\ts := append([]byte{65}, \"hi\"...)\n\treturn int(s[1])\n}",
		true, nil, []*Type{intT})
	h.funcs[fn.Name] = fn

	v, _ := evalOne(t, h, Config{}, parseExpr(t, `AppendUntyped()`))
	if got := v.Int64(); got != 104 {
		t.Fatalf("AppendUntyped() = %d, want 104 ('h')", got)
	}
}

func TestAppendTypedStringSpreadControl(t *testing.T) {
	intT := BasicType("int")

	h := newFakeHost(t)
	fn := parseFunc(t, "package p\nfunc AppendTyped() int {\n\ts := append([]byte{65}, string(\"hi\")...)\n\treturn int(s[2])\n}",
		true, nil, []*Type{intT})
	h.funcs[fn.Name] = fn

	v, _ := evalOne(t, h, Config{}, parseExpr(t, `AppendTyped()`))
	if got := v.Int64(); got != 105 {
		t.Fatalf("AppendTyped() = %d, want 105 ('i')", got)
	}
}

func TestSpreadTypedStringToIntRuneRejected(t *testing.T) {
	intT := BasicType("int")
	int32T := BasicType("int32")

	for _, tc := range []struct {
		name  string
		elem  *Type
		param string
	}{
		{"Int", intT, "int"},
		{"Rune", int32T, "rune"},
	} {
		h := newFakeHost(t)
		sliceT := &Type{Kind: KSlice, Elem: tc.elem}
		fn := parseFunc(t, "package p\nfunc VS"+tc.name+"(xs ..."+tc.param+") int { return len(xs) }",
			true, []*Type{sliceT}, []*Type{intT})
		fn.Variadic = true
		h.funcs[fn.Name] = fn

		cerr := evalErr(t, h, Config{}, parseExpr(t, fn.Name+`(string("hi")...)`))
		if cerr.Code != CodeEval {
			t.Fatalf("%s: code = %v, want CodeEval", tc.name, cerr.Code)
		}
	}
}

func TestSpreadTypedStringToOrdinaryByteRejected(t *testing.T) {
	intT := BasicType("int")
	uint8T := BasicType("uint8")
	sliceT := &Type{Kind: KSlice, Elem: uint8T}

	h := newFakeHost(t)
	fn := parseFunc(t, "package p\nfunc VByte(xs ...byte) int { return len(xs) }",
		true, []*Type{sliceT}, []*Type{intT})
	fn.Variadic = true
	h.funcs["VByte"] = fn

	cerr := evalErr(t, h, Config{}, parseExpr(t, `VByte(string("hi")...)`))
	if cerr.Code != CodeEval {
		t.Fatalf("code = %v, want CodeEval", cerr.Code)
	}
}
