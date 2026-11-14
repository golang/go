// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ctfe

import (
	"cmd/compile/internal/syntax"
	"go/ast"
	"go/constant"
	"go/parser"
	"go/token"
	"go/types"
	"math"
	"math/big"
	"strconv"
	"strings"
	"testing"
)

type fakeHost struct {
	t       *testing.T
	vars    map[string]Value
	funcs   map[string]*Func
	types   map[syntax.Expr]*Type
	imports map[string]bool
	pkgvars map[string]bool
}

func newFakeHost(t *testing.T) *fakeHost {
	return &fakeHost{
		t:       t,
		vars:    map[string]Value{},
		funcs:   map[string]*Func{},
		types:   map[syntax.Expr]*Type{},
		imports: map[string]bool{},
		pkgvars: map[string]bool{},
	}
}

func (h *fakeHost) Resolve(id *syntax.Name) Entity {
	if v, ok := h.vars[id.Value]; ok {
		return Entity{Kind: EntConst, Const: v}
	}
	if h.pkgvars[id.Value] {
		return Entity{Kind: EntPackageVar, Name: id.Value}
	}
	if f, ok := h.funcs[id.Value]; ok {
		if f.Const {
			return Entity{Kind: EntConstFunc, Fn: f}
		}
		return Entity{Kind: EntPlainFunc, Fn: f}
	}
	switch id.Value {
	case "len", "cap", "append", "copy", "make", "delete", "clear",
		"real", "imag", "complex", "print", "println", "panic", "recover", "new":
		return Entity{Kind: EntBuiltin, Builtin: id.Value}
	case "true", "false", "nil", "iota":
		return Entity{Kind: EntNone}
	}
	if isBasicName(id.Value) {
		return Entity{Kind: EntType}
	}
	return Entity{Kind: EntNone}
}

func (h *fakeHost) ResolveSel(sel *syntax.SelectorExpr) Entity {
	if x, ok := sel.X.(*syntax.Name); ok && h.imports[x.Value] {
		return Entity{Kind: EntImport, Name: x.Value + "." + sel.Sel.Value}
	}
	return Entity{Kind: EntNone}
}

func (h *fakeHost) TypeOf(e syntax.Expr) *Type {
	if t, ok := h.types[e]; ok {
		return t
	}
	if t := h.resolveTypeExpr(e); t != nil {
		return t
	}
	h.t.Fatalf("fakeHost.TypeOf: no type for %T %v", e, e)
	return nil
}

func (h *fakeHost) resolveTypeExpr(e syntax.Expr) *Type {
	switch e := syntax.Unparen(e).(type) {
	case *syntax.Name:
		if e.Value == "byte" {
			return BasicType("uint8")
		}
		if e.Value == "rune" {
			return BasicType("int32")
		}
		if isBasicName(e.Value) {
			return BasicType(e.Value)
		}
	case *syntax.SliceType:
		if el := h.resolveTypeExpr(e.Elem); el != nil {
			return &Type{Kind: KSlice, Elem: el}
		}
	case *syntax.ArrayType:
		el := h.resolveTypeExpr(e.Elem)
		if el == nil {
			return nil
		}
		if e.Len == nil {
			return &Type{Kind: KArray, Len: -1, Elem: el}
		}
		if lit, ok := syntax.Unparen(e.Len).(*syntax.BasicLit); ok {
			var n int64
			_, err := parseIntLit(lit.Value, &n)
			if err == nil {
				return &Type{Kind: KArray, Len: n, Elem: el}
			}
		}
	case *syntax.MapType:
		k := h.resolveTypeExpr(e.Key)
		v := h.resolveTypeExpr(e.Value)
		if k != nil && v != nil {
			return &Type{Kind: KMap, Key: k, Elem: v}
		}
	case *syntax.CallExpr:
		return h.resolveTypeExpr(e.Fun)
	case *syntax.CompositeLit:
		return h.resolveTypeExpr(e.Type)
	}
	return nil
}

func parseIntLit(s string, out *int64) (int64, error) {
	n, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		return 0, err
	}
	*out = n
	return n, nil
}

func isBasicName(name string) bool {
	switch name {
	case "bool", "string", "int", "int8", "int16", "int32", "int64",
		"uint", "uint8", "uint16", "uint32", "uint64", "uintptr",
		"float32", "float64", "complex64", "complex128", "any", "error":
		return true
	}
	return false
}

func parseFunc(t *testing.T, src string, constFunc bool, ptypes []*Type, rtypes []*Type) *Func {
	t.Helper()
	f, err := syntax.Parse(syntax.NewFileBase("test.lgo"), strings.NewReader(src), nil, nil, 0)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(f.DeclList) != 1 {
		t.Fatalf("want 1 decl, got %d", len(f.DeclList))
	}
	fd, ok := f.DeclList[0].(*syntax.FuncDecl)
	if !ok {
		t.Fatalf("want *FuncDecl, got %T", f.DeclList[0])
	}
	fn := &Func{Name: fd.Name.Value, Body: fd.Body, Const: constFunc}
	for i, p := range fd.Type.ParamList {
		if p.Name == nil {
			continue
		}
		var pt *Type
		if i < len(ptypes) {
			pt = ptypes[i]
		}
		fn.Params = append(fn.Params, Param{Decl: p.Name, Type: pt})
	}
	for i, r := range fd.Type.ResultList {
		var rt *Type
		if i < len(rtypes) {
			rt = rtypes[i]
		}
		fn.Results = append(fn.Results, rt)
		if r.Name != nil {
			fn.ResultNames = append(fn.ResultNames, r.Name.Value)
		} else {
			fn.ResultNames = append(fn.ResultNames, "")
		}
	}
	return fn
}

func parseExpr(t *testing.T, src string) syntax.Expr {
	t.Helper()
	f, err := syntax.Parse(syntax.NewFileBase("test.lgo"), strings.NewReader("package p\nvar _ = "+src+"\n"), nil, nil, 0)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	vd, ok := f.DeclList[0].(*syntax.VarDecl)
	if !ok {
		t.Fatalf("want *VarDecl, got %T", f.DeclList[0])
	}
	return vd.Values
}

func evalOne(t *testing.T, h *fakeHost, cfg Config, e syntax.Expr) (Value, Stats) {
	t.Helper()
	r, st, err := Eval(h, cfg, e)
	if err != nil {
		t.Fatalf("Eval: %v", err)
	}
	return r, st
}

func evalErr(t *testing.T, h *fakeHost, cfg Config, e syntax.Expr) *Error {
	t.Helper()
	_, _, err := Eval(h, cfg, e)
	if err == nil {
		t.Fatalf("Eval succeeded, want error")
	}
	cerr, ok := err.(*Error)
	if !ok {
		t.Fatalf("want *Error, got %T (%v)", err, err)
	}
	return cerr
}

func TestWidths(t *testing.T) {
	for _, word := range []int64{4, 8} {
		h := newFakeHost(t)
		cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: word}
		conv := parseExpr(t, `int(1 << 32)`)
		h.types[conv] = BasicType("int")
		if word == 8 {
			v, _ := evalOne(t, h, cfg, conv)
			if v.Int64() != 4294967296 {
				t.Fatalf("word %d: got %v, want 4294967296", word, v)
			}
			continue
		}
		cerr := evalErr(t, h, cfg, conv)
		if cerr.Code != CodeEval {
			t.Fatalf("word %d: code = %v, want CodeEval", word, cerr.Code)
		}
	}

	h := newFakeHost(t)
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	ok := parseExpr(t, `int8(127)`)
	h.types[ok] = BasicType("int8")
	v, _ := evalOne(t, h, cfg, ok)
	if v.Int64() != 127 {
		t.Fatalf("got %v, want 127", v)
	}
	bad := parseExpr(t, `int8(128)`)
	h.types[bad] = BasicType("int8")
	if cerr := evalErr(t, h, cfg, bad); cerr.Code != CodeEval {
		t.Fatalf("code = %v, want CodeEval", cerr.Code)
	}

	fn := parseFunc(t, `package p
func Wrap(x int) int8 { return int8(x) }`, false, []*Type{BasicType("int")}, []*Type{BasicType("int8")})
	h.funcs["Wrap"] = fn
	call := parseExpr(t, `Wrap(200)`)
	h.types[call] = BasicType("int8")
	v, _ = evalOne(t, h, cfg, call)
	if v.Int64() != -56 {
		t.Fatalf("got %v, want -56", v)
	}
}

func TestWidthsComposite(t *testing.T) {
	for _, word := range []int64{4, 8} {
		h := newFakeHost(t)
		cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: word}
		lit := parseExpr(t, `[]int{1 << 32}`)
		if word == 8 {
			v, _ := evalOne(t, h, cfg, lit)
			elems := sliceElems(v)
			if len(elems) != 1 {
				t.Fatalf("word %d: %d elements, want 1", word, len(elems))
			}
			if elems[0].Int64() != 4294967296 {
				t.Fatalf("word %d: element = %v, want 4294967296", word, elems[0])
			}
			continue
		}
		cerr := evalErr(t, h, cfg, lit)
		if cerr.Code != CodeEval {
			t.Fatalf("word %d: code = %v, want CodeEval", word, cerr.Code)
		}
	}
}

func TestStepBoundaries(t *testing.T) {
	h := newFakeHost(t)
	fn := parseFunc(t, `package p
func Burn(n int) int {
	total := 0
	for i := 0; i < n; i++ {
		total++
	}
	return total
}`, true, []*Type{BasicType("int")}, []*Type{BasicType("int")})
	h.funcs["Burn"] = fn

	measure := func(src string) (Value, int) {
		hh := newFakeHost(t)
		hh.funcs["Burn"] = fn
		call := parseExpr(t, src)
		v, st, err := Eval(hh, Config{MaxSteps: 1 << 30, MaxValueBytes: 1 << 20, WordSize: 8}, call)
		if err != nil {
			t.Fatalf("measure %s: %v", src, err)
		}
		return v, st.Steps
	}
	v2, steps2 := measure(`Burn(2)`)
	if v2.Int64() != 2 {
		t.Fatalf("Burn(2) = %v, want 2", v2)
	}
	if steps2 <= 0 {
		t.Fatalf("non-positive step count %d", steps2)
	}
	at := func(limit int) *Error {
		hh := newFakeHost(t)
		hh.funcs["Burn"] = fn
		call := parseExpr(t, `Burn(2)`)
		_, _, err := Eval(hh, Config{MaxSteps: limit, MaxValueBytes: 1 << 20, WordSize: 8}, call)
		if err == nil {
			return nil
		}
		cerr, ok := err.(*Error)
		if !ok {
			t.Fatalf("want *Error, got %T", err)
		}
		return cerr
	}
	if cerr := at(steps2 - 1); cerr == nil || cerr.Code != CodeStepLimit {
		t.Fatalf("limit N-1 err = %v, want step limit", cerr)
	}
	if cerr := at(steps2); cerr != nil {
		t.Fatalf("limit N err = %v, want success", cerr)
	}
	if cerr := at(steps2 + 1); cerr != nil {
		t.Fatalf("limit N+1 err = %v, want success", cerr)
	}
}

func TestValueSizeBoundaries(t *testing.T) {
	h := newFakeHost(t)
	small := Config{MaxSteps: 100000, MaxValueBytes: 10, WordSize: 8}
	arr10 := parseExpr(t, `[10]byte{}`)
	h.types[arr10] = &Type{Kind: KArray, Len: 10, Elem: BasicType("uint8")}
	if _, _, err := Eval(h, small, arr10); err != nil {
		t.Fatalf("[10]byte{} with 10-byte cap: %v", err)
	}
	arr11 := parseExpr(t, `[11]byte{}`)
	h.types[arr11] = &Type{Kind: KArray, Len: 11, Elem: BasicType("uint8")}
	if cerr := evalErr(t, h, small, arr11); cerr.Code != CodeTooLarge {
		t.Fatalf("code = %v, want CodeTooLarge", cerr.Code)
	}

	def := Config{}
	mega := parseExpr(t, `[1048576]byte{}`)
	h.types[mega] = &Type{Kind: KArray, Len: 1048576, Elem: BasicType("uint8")}
	if _, _, err := Eval(h, def, mega); err != nil {
		t.Fatalf("1 MB array: %v", err)
	}
	mega1 := parseExpr(t, `[1048577]byte{}`)
	h.types[mega1] = &Type{Kind: KArray, Len: 1048577, Elem: BasicType("uint8")}
	if cerr := evalErr(t, h, def, mega1); cerr.Code != CodeTooLarge {
		t.Fatalf("code = %v, want CodeTooLarge", cerr.Code)
	}
}

func TestFloatMapKeys(t *testing.T) {
	h := newFakeHost(t)
	neg := parseFunc(t, `package p
func NegZero(x float64) float64 {
	z := x - x
	return -z
}`, true, []*Type{BasicType("float64")}, []*Type{BasicType("float64")})
	h.funcs["NegZero"] = neg
	frac := parseFunc(t, `package p
func Frac(x float64) float64 {
	z := x - x
	return z / z
}`, true, []*Type{BasicType("float64")}, []*Type{BasicType("float64")})
	h.funcs["Frac"] = frac
	cfg := Config{}

	dup := parseExpr(t, `map[float64]string{0.0: "a", NegZero(1.0): "b"}`)
	h.types[dup] = &Type{Kind: KMap, Key: BasicType("float64"), Elem: UntypedString()}
	if cerr := evalErr(t, h, cfg, dup); cerr.Code != CodeEval || !strings.Contains(cerr.Detail, "duplicate") {
		t.Fatalf("dup err = %v, want duplicate CodeEval", cerr)
	}
	nan := parseExpr(t, `map[float64]string{Frac(1.0): "nan"}`)
	h.types[nan] = &Type{Kind: KMap, Key: BasicType("float64"), Elem: UntypedString()}
	if cerr := evalErr(t, h, cfg, nan); cerr.Code != CodeEval || !strings.Contains(cerr.Detail, "NaN") {
		t.Fatalf("nan err = %v, want NaN CodeEval", cerr)
	}
	ok := parseExpr(t, `map[float64]string{NegZero(1.0): "neg"}`)
	h.types[ok] = &Type{Kind: KMap, Key: BasicType("float64"), Elem: UntypedString()}
	v, _ := evalOne(t, h, cfg, ok)
	if v.mapLen() != 1 {
		t.Fatalf("map len = %d, want 1", v.mapLen())
	}
	if got := v.mapKeyBits(0); got != 0 || math.Signbit(math.Float64frombits(got)) {
		t.Fatalf("key bits = %x, want +0.0", got)
	}
}

func TestComplexMapKeys(t *testing.T) {
	h := newFakeHost(t)
	neg := parseFunc(t, `package p
func CNegZero(x complex128) complex128 {
	z := x - x
	return -z
}`, true, []*Type{BasicType("complex128")}, []*Type{BasicType("complex128")})
	h.funcs["CNegZero"] = neg
	frac := parseFunc(t, `package p
func CFrac(x complex128) complex128 {
	z := x - x
	return z / z
}`, true, []*Type{BasicType("complex128")}, []*Type{BasicType("complex128")})
	h.funcs["CFrac"] = frac
	cfg := Config{}

	dup := parseExpr(t, `map[complex128]string{complex(0, 0): "a", CNegZero(complex(1, 1)): "b"}`)
	h.types[dup] = &Type{Kind: KMap, Key: BasicType("complex128"), Elem: UntypedString()}
	if cerr := evalErr(t, h, cfg, dup); cerr.Code != CodeEval || !strings.Contains(cerr.Detail, "duplicate") {
		t.Fatalf("dup err = %v, want duplicate CodeEval", cerr)
	}
	nan := parseExpr(t, `map[complex128]string{CFrac(complex(1, 1)): "nan"}`)
	h.types[nan] = &Type{Kind: KMap, Key: BasicType("complex128"), Elem: UntypedString()}
	if cerr := evalErr(t, h, cfg, nan); cerr.Code != CodeEval || !strings.Contains(cerr.Detail, "NaN") {
		t.Fatalf("nan err = %v, want NaN CodeEval", cerr)
	}
	ok := parseExpr(t, `map[complex128]string{CNegZero(complex(1, 1)): "neg"}`)
	h.types[ok] = &Type{Kind: KMap, Key: BasicType("complex128"), Elem: UntypedString()}
	v, _ := evalOne(t, h, cfg, ok)
	if v.mapLen() != 1 {
		t.Fatalf("map len = %d, want 1", v.mapLen())
	}
	keys, _ := v.MapEntries()
	c := keys[0].Complex()
	if real(c) != 0 || imag(c) != 0 || math.Signbit(real(c)) || math.Signbit(imag(c)) {
		t.Fatalf("key = %v, want (+0,+0)", c)
	}
}

func TestCompositeNaNMapKeys(t *testing.T) {
	floatType := BasicType("float64")
	complexType := BasicType("complex128")
	arrayOf := func(v Value) Value {
		return Value{typ: &Type{Kind: KArray, Len: 2, Elem: v.typ}, array: []Value{zeroValue(v.typ), v}}
	}
	structOf := func(v Value) Value {
		return Value{typ: &Type{Kind: KStruct, Fields: []Field{{Name: "V", Type: v.typ}}}, strct: []Value{v}}
	}
	for _, shape := range []struct {
		name string
		key  func(Value) Value
	}{
		{"array", arrayOf},
		{"struct", structOf},
		{"arrayStructArray", func(v Value) Value { return arrayOf(structOf(arrayOf(v))) }},
		{"structArrayStruct", func(v Value) Value { return structOf(arrayOf(structOf(v))) }},
	} {
		for _, leaf := range []struct {
			name string
			bad  Value
			good Value
		}{
			{"float", FloatValue(math.NaN(), floatType), FloatValue(2, floatType)},
			{"complexReal", ComplexValue(complex(math.NaN(), 1), complexType), ComplexValue(complex(2, 1), complexType)},
			{"complexImag", ComplexValue(complex(1, math.NaN()), complexType), ComplexValue(complex(1, 2), complexType)},
		} {
			t.Run(shape.name+"/"+leaf.name+"/NaN", func(t *testing.T) {
				testCompositeMapKey(t, shape.key(leaf.bad), "NaN is not a valid const map key")
			})
			t.Run(shape.name+"/"+leaf.name+"/finite", func(t *testing.T) {
				testCompositeMapKey(t, shape.key(leaf.good), "")
			})
		}
	}
}

func testCompositeMapKey(t *testing.T, key Value, wantError string) {
	t.Helper()
	h := newFakeHost(t)
	h.vars["key"] = key
	mapType := &Type{Kind: KMap, Key: key.typ, Elem: BasicType("int")}
	lit := parseExpr(t, `map[Key]int{key: 7}`).(*syntax.CompositeLit)
	h.types[lit] = mapType
	h.types[lit.Type] = mapType
	h.types[lit.Type.(*syntax.MapType).Key] = key.typ
	h.vars["m"] = Value{typ: mapType, mval: &mapVal{
		keys: []Value{key},
		vals: []Value{IntValue(big.NewInt(7), BasicType("int"))},
	}}
	for _, expr := range []syntax.Expr{lit, parseExpr(t, `m[key]`)} {
		if wantError != "" {
			cerr := evalErr(t, h, Config{}, expr)
			if cerr.Code != CodeEval || cerr.Detail != wantError {
				t.Fatalf("error = %v, want %s CodeEval", cerr, wantError)
			}
			continue
		}
		v, _ := evalOne(t, h, Config{}, expr)
		if expr == lit && v.mapLen() != 1 {
			t.Fatalf("map len = %d, want 1", v.mapLen())
		}
		if expr != lit && v.Int64() != 7 {
			t.Fatalf("lookup = %v, want 7", v)
		}
	}
}

func TestAbsentMapKey(t *testing.T) {
	h := newFakeHost(t)
	mk := &Type{Kind: KMap, Key: UntypedString(), Elem: BasicType("int")}
	mv := Value{typ: mk, mval: &mapVal{
		keys: []Value{StringValue("a", UntypedString())},
		vals: []Value{IntValue(big.NewInt(1), BasicType("int"))},
	}}
	h.vars["m"] = mv
	cfg := Config{}
	idx := parseExpr(t, `m["zzz"]`)
	if cerr := evalErr(t, h, cfg, idx); cerr.Code != CodeEval || !strings.Contains(cerr.Detail, "key not present") {
		t.Fatalf("absent err = %v, want key-not-present CodeEval", cerr)
	}
	hit := parseExpr(t, `m["a"]`)
	v, _ := evalOne(t, h, cfg, hit)
	if v.Int64() != 1 {
		t.Fatalf("got %v, want 1", v)
	}
}

func TestDynamicImpurity(t *testing.T) {
	h := newFakeHost(t)
	h.imports["fmt"] = true
	cfg := Config{}

	call := parseExpr(t, `fmt.Println("x")`)
	if cerr := evalErr(t, h, cfg, call); cerr.Code != CodeEval {
		t.Fatalf("code = %v, want CodeEval", cerr.Code)
	}
	rec := parseExpr(t, `recover()`)
	if cerr := evalErr(t, h, cfg, rec); cerr.Code != CodeEval {
		t.Fatalf("code = %v, want CodeEval", cerr.Code)
	}

	bad := parseFunc(t, `package p
func Bad() int {
	println("x")
	return 1
}`, true, nil, []*Type{BasicType("int")})
	h.funcs["Bad"] = bad
	top := parseExpr(t, `Bad()`)
	if cerr := evalErr(t, h, cfg, top); cerr.Code != CodeSilent {
		t.Fatalf("code = %v, want CodeSilent", cerr.Code)
	}
}

func TestEvalDefaultStepBudgetBoundary(t *testing.T) {
	fn := parseFunc(t, `package p
func Burn1(n int) int {
	total := 0
	for i := 0; i < n; i++ {
		total++
	}
	total++
	return total
}`, true, []*Type{BasicType("int")}, []*Type{BasicType("int")})
	measure := func(n int) (int64, int) {
		h := newFakeHost(t)
		h.funcs["Burn1"] = fn
		v, st, err := Eval(h, Config{MaxSteps: 1 << 30, MaxValueBytes: 1 << 20, WordSize: 8}, parseExpr(t, `Burn1(`+strconv.Itoa(n)+`)`))
		if err != nil {
			t.Fatalf("measure Burn1(%d): %v", n, err)
		}
		return v.Int64(), st.Steps
	}
	_, base := measure(0)
	_, one := measure(1)
	_, two := measure(2)
	cost := one - base
	if cost <= 0 || two != base+2*cost {
		t.Fatalf("non-linear step cost %d, %d, %d", base, one, two)
	}
	n := (100000 - base) / cost
	if n < 0 || (100000-base)%cost != 0 {
		t.Fatalf("budget 100000 unreachable from base %d cost %d", base, cost)
	}
	h := newFakeHost(t)
	h.funcs["Burn1"] = fn
	v, st, err := Eval(h, Config{}, parseExpr(t, `Burn1(`+strconv.Itoa(n)+`)`))
	if err != nil {
		t.Fatalf("Burn1(%d) with zero Config: %v", n, err)
	}
	if v.Int64() != int64(n)+1 {
		t.Fatalf("Burn1(%d) = %v, want %d", n, v.Int64(), n+1)
	}
	if st.Steps != 100000 {
		t.Fatalf("Burn1(%d) steps = %d, want exactly 100000", n, st.Steps)
	}
	over := newFakeHost(t)
	over.funcs["Burn1"] = fn
	_, _, err = Eval(over, Config{}, parseExpr(t, `Burn1(`+strconv.Itoa(n+1)+`)`))
	cerr, ok := err.(*Error)
	if !ok || cerr.Code != CodeStepLimit {
		t.Fatalf("Burn1(%d) with zero Config err = %v, want step limit", n+1, err)
	}
}

func TestExactComplexConstants(t *testing.T) {
	for _, tt := range []struct {
		name string
		expr string
		want bool
	}{
		{"imaginarySubtraction", `9007199254740993i - 9007199254740992i == 1i`, true},
		{"product", `9007199254740993i * 1i + 9007199254740992 == -1`, true},
		{"sum", `(9007199254740993 + 1i) - 9007199254740992 == 1 + 1i`, true},
		{"quotient", `1i / 3 == 0.3333333333333333i`, false},
		{"negation", `-9007199254740993i + 9007199254740992i == -1i`, true},
		{"decimalSum", `0.1i + 0.2i == 0.3i`, true},
		{"inequality", `9007199254740993i == 9007199254740992i`, false},
		{"floatOperand", `9007199254740993.0 + 1i - 9007199254740992.0 == 1 + 1i`, true},
		{"runeOperand", `(('a' << 53) + 1) + 1i - ('a' << 53) == 1 + 1i`, true},
		{"complexBuiltin", `complex(9007199254740993, 0.1) - complex(9007199254740992, 0) == 1 + 0.1i`, true},
		{"hexadecimal", `0x20000000000001p0i - 0x1p53i == 1i`, true},
		{"underscores", `9_007_199_254_740_993i - 9_007_199_254_740_992i == 1i`, true},
		{"decimalLeadingZero", `0123i == 123i && 08i == 8i`, true},
		{"nonzeroTinyDivisor", `1e-1000i / 1e-1000i == 1`, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			v, _ := evalOne(t, newFakeHost(t), Config{}, parseExpr(t, tt.expr))
			if v.Bool() != tt.want {
				t.Fatalf("%s = %v, want %v", tt.expr, v.Bool(), tt.want)
			}
		})
	}
}

func TestComplexEqualityCoercion(t *testing.T) {
	for _, tt := range []struct {
		name string
		typ  string
		expr string
		arg  string
		want bool
	}{
		{"complex64Right", "complex64", "x == 0.1", "complex64(0.1)", true},
		{"complex64Left", "complex64", "0.1 == x", "complex64(0.1)", true},
		{"complex64ImaginaryRight", "complex64", "x == 0.1 + 0.2i", "complex64(0.1 + 0.2i)", true},
		{"complex64ImaginaryLeft", "complex64", "0.1 + 0.2i == x", "complex64(0.1 + 0.2i)", true},
		{"complex128Right", "complex128", "x == 0.1", "complex128(complex64(0.1))", false},
		{"complex128Left", "complex128", "0.1 == x", "complex128(complex64(0.1))", false},
		{"typedComplex128", "complex128", "x == complex128(0.1)", "complex128(complex64(0.1))", false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			h := newFakeHost(t)
			h.funcs["Equal"] = parseFunc(t, "package p\nfunc Equal(x "+tt.typ+") bool { return "+tt.expr+" }",
				true, []*Type{BasicType(tt.typ)}, []*Type{BasicType("bool")})
			v, _ := evalOne(t, h, Config{}, parseExpr(t, "Equal("+tt.arg+")"))
			if v.Bool() != tt.want {
				t.Fatalf("Equal(%s) with %s = %v, want %v", tt.arg, tt.expr, v.Bool(), tt.want)
			}
		})
	}
}

func TestExactComplexRendering(t *testing.T) {
	for _, tt := range []struct {
		name  string
		expr  string
		check string
	}{
		{"largeInteger", `9007199254740993 + 1i / 3`, `(res - 9007199254740992) * 3 == 3 + 1i`},
		{"negativeRational", `-1.0 / 3 - 9007199254740993i`, `(res + 9007199254740992i) * 3 == -1 - 3i`},
	} {
		t.Run(tt.name, func(t *testing.T) {
			h := newFakeHost(t)
			v, _ := evalOne(t, h, Config{}, parseExpr(t, tt.expr))
			rendered, err := v.Expr(syntax.Pos{})
			if err != nil {
				t.Fatalf("Expr: %v", err)
			}
			h.vars["res"], _ = evalOne(t, h, Config{}, rendered)
			eq, _ := evalOne(t, h, Config{}, parseExpr(t, tt.check))
			if !eq.Bool() {
				t.Fatalf("rendering %s did not preserve its exact value", tt.expr)
			}
		})
	}
}

func TestComplexRenderingShadowedBuiltin(t *testing.T) {
	for _, tt := range []struct {
		name string
		expr string
		typ  string
	}{
		{"pureImaginary", `1i`, ""},
		{"negativeImaginary", `2 - 3i`, ""},
		{"exactRationals", `1.0/3 - 9007199254740993i`, ""},
		{"negativeRationals", `-1.0/3 - 2i/7`, ""},
		{"complex128", `complex128(0.1 - 0.2i)`, "complex128"},
		{"complex64", `complex64(0.1 - 0.2i)`, "complex64"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			v, _ := evalOne(t, newFakeHost(t), Config{}, parseExpr(t, tt.expr))
			rendered, err := v.Expr(syntax.Pos{})
			if err != nil {
				t.Fatalf("Expr: %v", err)
			}
			syntax.Inspect(rendered, func(n syntax.Node) bool {
				if call, ok := n.(*syntax.CallExpr); ok {
					if name, ok := call.Fun.(*syntax.Name); ok && name.Value == "complex" {
						t.Error("rendered expression calls the shadowable complex builtin")
					}
				}
				return true
			})
			var text strings.Builder
			if _, err := syntax.Fprint(&text, rendered, syntax.LineForm); err != nil {
				t.Fatalf("Fprint: %v", err)
			}
			expr := text.String()
			if tt.typ != "" {
				expr = tt.typ + "(" + expr + ")"
			}
			src := "package p\nvar complex = 7\nconst got = " + expr + "\nconst want = " + tt.expr
			fset := token.NewFileSet()
			file, err := parser.ParseFile(fset, "render.go", src, 0)
			if err != nil {
				t.Fatalf("parse rendered expression: %v", err)
			}
			conf := types.Config{}
			pkg, err := conf.Check("p", fset, []*ast.File{file}, nil)
			if err != nil {
				t.Fatalf("type-check rendered expression %s: %v", expr, err)
			}
			got := pkg.Scope().Lookup("got").(*types.Const)
			want := pkg.Scope().Lookup("want").(*types.Const)
			if !types.Identical(got.Type(), want.Type()) {
				t.Errorf("rendered type = %s, want %s", got.Type(), want.Type())
			}
			if !constant.Compare(got.Val(), token.EQL, want.Val()) {
				t.Errorf("rendered value = %s, want %s", got.Val().ExactString(), want.Val().ExactString())
			}
		})
	}
}

func TestExactComplexDivisionByZero(t *testing.T) {
	for _, expr := range []string{`1i / (1i - 1i)`, `1i / (1e-1000i - 1e-1000i)`} {
		cerr := evalErr(t, newFakeHost(t), Config{}, parseExpr(t, expr))
		if cerr.Code != CodeEval || !strings.Contains(cerr.Detail, "division by zero") {
			t.Fatalf("%s: error = %v, want division by zero CodeEval", expr, cerr)
		}
	}
}
