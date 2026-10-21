// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ctfe

// Unit tests for the M5-S2 compile-time interpreter (SPEC Sec 2.1,
// T-201). The fake host below resolves a closed set of expression
// shapes, so every test pins interpreter behavior without a type
// checker: target integer widths (#102), step-limit boundaries,
// the 1 MB value cap, -0.0 canonicalization and NaN rejection (#103),
// and dynamic impurity reporting.

import (
	"cmd/compile/internal/syntax"
	"math"
	"math/big"
	"strconv"
	"strings"
	"testing"
)

// fakeHost resolves basic types, untyped literals, a fixed name
// environment, and caller-provided const funcs.
type fakeHost struct {
	t       *testing.T
	word    int64
	vars    map[string]Value
	funcs   map[string]*Func
	types   map[syntax.Expr]*Type
	imports map[string]bool
	pkgvars map[string]bool
}

func newFakeHost(t *testing.T, word int64) *fakeHost {
	return &fakeHost{
		t:       t,
		word:    word,
		vars:    map[string]Value{},
		funcs:   map[string]*Func{},
		types:   map[syntax.Expr]*Type{},
		imports: map[string]bool{},
		pkgvars: map[string]bool{},
	}
}

func (h *fakeHost) WordSize() int64 { return h.word }

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
		return Entity{Kind: EntType, Type: BasicType(id.Value, h.word)}
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

// resolveTypeExpr resolves type expressions structurally, mirroring
// the real adapter: basic names, slices, arrays (with literal lengths,
// -1 for ...), maps, conversions by their target, and composite
// literals by their type.
func (h *fakeHost) resolveTypeExpr(e syntax.Expr) *Type {
	switch e := syntax.Unparen(e).(type) {
	case *syntax.Name:
		if e.Value == "byte" {
			return BasicType("uint8", h.word)
		}
		if e.Value == "rune" {
			return BasicType("int32", h.word)
		}
		if isBasicName(e.Value) {
			return BasicType(e.Value, h.word)
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

// parseIntLit parses a decimal integer literal.
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
		"float32", "float64", "any", "error":
		return true
	}
	return false
}

// parseFunc parses one `func` declaration and builds a Func with the
// given const flag, parameter types, and result types.
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

// parseExpr parses one expression.
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

// evalOne evaluates e and requires exactly one value.
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

// TestWidths pins target integer widths (#102): 1<<32 converts to int
// on 64-bit targets and overflows on 32-bit targets; int8 boundaries
// hold; variable values wrap while constants overflow.
func TestWidths(t *testing.T) {
	for _, word := range []int64{4, 8} {
		h := newFakeHost(t, word)
		cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: word}
		conv := parseExpr(t, `int(1 << 32)`)
		h.types[conv] = BasicType("int", word)
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

	h := newFakeHost(t, 8)
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	ok := parseExpr(t, `int8(127)`)
	h.types[ok] = BasicType("int8", 8)
	v, _ := evalOne(t, h, cfg, ok)
	if v.Int64() != 127 {
		t.Fatalf("got %v, want 127", v)
	}
	bad := parseExpr(t, `int8(128)`)
	h.types[bad] = BasicType("int8", 8)
	if cerr := evalErr(t, h, cfg, bad); cerr.Code != CodeEval {
		t.Fatalf("code = %v, want CodeEval", cerr.Code)
	}

	// A variable int8 wraps instead of overflowing: Wrap(200) is -56.
	fn := parseFunc(t, `package p
func Wrap(x int) int8 { return int8(x) }`, false, []*Type{BasicType("int", 8)}, []*Type{BasicType("int8", 8)})
	h.funcs["Wrap"] = fn
	call := parseExpr(t, `Wrap(200)`)
	h.types[call] = BasicType("int8", 8)
	v, _ = evalOne(t, h, cfg, call)
	if v.Int64() != -56 {
		t.Fatalf("got %v, want -56", v)
	}
}

// TestWidthsComposite pins target integer widths inside const composites
// (#180): a composite holding 1<<32 evaluates with the target width, so
// the 64-bit target keeps the element while the 32-bit target reports a
// compile-time overflow. The scalar TestWidths above misses an
// element-width bug that only fires inside literals.
func TestWidthsComposite(t *testing.T) {
	for _, word := range []int64{4, 8} {
		h := newFakeHost(t, word)
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

// TestStepBoundaries checks the step limit at N-1, N, N+1 with a small
// limit, measuring exact step counts first so the test stays aligned
// with the step model.
func TestStepBoundaries(t *testing.T) {
	h := newFakeHost(t, 8)
	fn := parseFunc(t, `package p
func Burn(n int) int {
	total := 0
	for i := 0; i < n; i++ {
		total++
	}
	return total
}`, true, []*Type{BasicType("int", 8)}, []*Type{BasicType("int", 8)})
	h.funcs["Burn"] = fn

	// Measure exact step counts with no effective limit, then check
	// the boundary at N-1, N, N+1 around a measured count.
	measure := func(src string) (Value, int) {
		hh := newFakeHost(t, 8)
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
		hh := newFakeHost(t, 8)
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

// TestValueSizeBoundaries checks the 1 MB cap at N-1, N, N+1 with a
// small limit, plus the default-limit boundaries.
func TestValueSizeBoundaries(t *testing.T) {
	h := newFakeHost(t, 8)
	small := Config{MaxSteps: 100000, MaxValueBytes: 10, WordSize: 8}
	arr10 := parseExpr(t, `[10]byte{}`)
	h.types[arr10] = &Type{Kind: KArray, Len: 10, Elem: BasicType("uint8", 8)}
	if _, _, err := Eval(h, small, arr10); err != nil {
		t.Fatalf("[10]byte{} with 10-byte cap: %v", err)
	}
	arr11 := parseExpr(t, `[11]byte{}`)
	h.types[arr11] = &Type{Kind: KArray, Len: 11, Elem: BasicType("uint8", 8)}
	if cerr := evalErr(t, h, small, arr11); cerr.Code != CodeTooLarge {
		t.Fatalf("code = %v, want CodeTooLarge", cerr.Code)
	}

	def := DefaultConfig()
	mega := parseExpr(t, `[1048576]byte{}`)
	h.types[mega] = &Type{Kind: KArray, Len: 1048576, Elem: BasicType("uint8", 8)}
	if _, _, err := Eval(h, def, mega); err != nil {
		t.Fatalf("1 MB array: %v", err)
	}
	mega1 := parseExpr(t, `[1048577]byte{}`)
	h.types[mega1] = &Type{Kind: KArray, Len: 1048577, Elem: BasicType("uint8", 8)}
	if cerr := evalErr(t, h, def, mega1); cerr.Code != CodeTooLarge {
		t.Fatalf("code = %v, want CodeTooLarge", cerr.Code)
	}
}

// TestFloatMapKeys checks -0.0 canonicalization and NaN rejection (#103).
func TestFloatMapKeys(t *testing.T) {
	h := newFakeHost(t, 8)
	neg := parseFunc(t, `package p
func NegZero(x float64) float64 {
	z := x - x
	return -z
}`, true, []*Type{BasicType("float64", 8)}, []*Type{BasicType("float64", 8)})
	h.funcs["NegZero"] = neg
	frac := parseFunc(t, `package p
func Frac(x float64) float64 {
	z := x - x
	return z / z
}`, true, []*Type{BasicType("float64", 8)}, []*Type{BasicType("float64", 8)})
	h.funcs["Frac"] = frac
	cfg := DefaultConfig()

	dup := parseExpr(t, `map[float64]string{0.0: "a", NegZero(1.0): "b"}`)
	h.types[dup] = &Type{Kind: KMap, Key: BasicType("float64", 8), Elem: UntypedString()}
	if cerr := evalErr(t, h, cfg, dup); cerr.Code != CodeEval || !strings.Contains(cerr.Detail, "duplicate") {
		t.Fatalf("dup err = %v, want duplicate CodeEval", cerr)
	}
	nan := parseExpr(t, `map[float64]string{Frac(1.0): "nan"}`)
	h.types[nan] = &Type{Kind: KMap, Key: BasicType("float64", 8), Elem: UntypedString()}
	if cerr := evalErr(t, h, cfg, nan); cerr.Code != CodeEval || !strings.Contains(cerr.Detail, "NaN") {
		t.Fatalf("nan err = %v, want NaN CodeEval", cerr)
	}
	ok := parseExpr(t, `map[float64]string{NegZero(1.0): "neg"}`)
	h.types[ok] = &Type{Kind: KMap, Key: BasicType("float64", 8), Elem: UntypedString()}
	v, _ := evalOne(t, h, cfg, ok)
	if v.MapLen() != 1 {
		t.Fatalf("map len = %d, want 1", v.MapLen())
	}
	if got := v.MapKeyBits(0); got != 0 || math.Signbit(math.Float64frombits(got)) {
		t.Fatalf("key bits = %x, want +0.0", got)
	}
}

// TestAbsentMapKey checks that a missing key in a single-value const
// map read fails instead of guessing a value.
func TestAbsentMapKey(t *testing.T) {
	h := newFakeHost(t, 8)
	mk := &Type{Kind: KMap, Key: UntypedString(), Elem: BasicType("int", 8)}
	mv := Value{typ: mk, mval: &mapVal{
		keys: []Value{StringValue("a", UntypedString())},
		vals: []Value{IntValue(big.NewInt(1), BasicType("int", 8))},
	}}
	h.vars["m"] = mv
	cfg := DefaultConfig()
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

// TestDynamicImpurity checks evaluation-time impurity: imported calls
// and recover fail, while an impurity inside a const func body aborts
// silently (the static purity check reports it).
func TestDynamicImpurity(t *testing.T) {
	h := newFakeHost(t, 8)
	h.imports["fmt"] = true
	cfg := DefaultConfig()

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
}`, true, nil, []*Type{BasicType("int", 8)})
	h.funcs["Bad"] = bad
	top := parseExpr(t, `Bad()`)
	if cerr := evalErr(t, h, cfg, top); cerr.Code != CodeSilent {
		t.Fatalf("code = %v, want CodeSilent", cerr.Code)
	}
}
