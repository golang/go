// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ctfe

// This file implements M5-S2 value constructors, conversions, composite
// literals, indexing, slicing, builtins, and size accounting (SPEC Sec
// 2.1, T-201).

import (
	"cmd/compile/internal/syntax"
	"fmt"
	"math"
	"math/big"
	"slices"
	"strconv"
	"unicode/utf8"
)

// Constructors.

func intValue(n *big.Int, t *Type, cst bool) Value {
	return Value{typ: t, constant: cst, ival: n}
}

func byteValue(b byte) Value {
	return Value{typ: &Type{Kind: KUint8}, constant: false, ival: big.NewInt(int64(b))}
}

func runeValue(r rune) Value {
	return Value{typ: &Type{Kind: KInt32}, constant: false, ival: big.NewInt(int64(r))}
}

func sliceValue(elem *Type, elems []Value, cap_ int) Value {
	backing := append([]Value(nil), elems...)
	return Value{
		typ:      &Type{Kind: KSlice, Elem: elem},
		slice:    &sliceVal{backing: &backing, length: len(elems), cap_: cap_},
		constant: false,
	}
}

func sliceElems(v Value) []Value {
	return (*v.slice.backing)[v.slice.lo : v.slice.lo+v.slice.length]
}

// sliceLen counts set slice elements.
func sliceLen(v Value) int { return v.slice.length }

// BoolValue builds a boolean value.
func BoolValue(b bool, t *Type) Value { return Value{typ: t, constant: true, bval: b} }

// StringValue builds a string value.
func StringValue(s string, t *Type) Value { return Value{typ: t, constant: true, sval: s} }

// IntValue builds an integer value.
func IntValue(n *big.Int, t *Type) Value { return Value{typ: t, constant: true, ival: n} }

// FloatValue builds a float value.
func FloatValue(f float64, t *Type) Value { return Value{typ: t, constant: true, f64: f} }

// defaulted maps an untyped value to its default type, leaving typed
// values alone.
func defaulted(v Value) Value {
	if v.typ == nil || !isUntyped(v.typ.Kind) {
		return v
	}
	v.typ = defaultType(v.typ.Kind, 8)
	return v
}

// zeroValue returns the zero value of t.
func zeroValue(t *Type) Value {
	switch t.Kind {
	case KBool, KUntypedBool:
		return Value{typ: &Type{Kind: KBool}, bval: false}
	case KString, KUntypedString:
		return Value{typ: &Type{Kind: KString}, sval: ""}
	case KFloat32, KFloat64, KUntypedFloat:
		k := t.Kind
		if k == KUntypedFloat {
			k = KFloat64
		}
		return Value{typ: &Type{Kind: k}, f64: 0}
	case KSlice, KMap:
		return Value{typ: t, isNil: true}
	case KArray:
		elems := make([]Value, t.Len)
		for i := range elems {
			elems[i] = zeroValue(t.Elem)
		}
		return Value{typ: t, array: elems}
	case KStruct:
		fields := make([]Value, len(t.Fields))
		for i, f := range t.Fields {
			fields[i] = zeroValue(f.Type)
		}
		return Value{typ: t, strct: fields}
	}
	if isInteger(t.Kind) {
		k := t.Kind
		if isUntyped(k) {
			k = KInt
		}
		if k == KUntypedRune {
			k = KInt32
		}
		return Value{typ: &Type{Kind: k}, ival: new(big.Int)}
	}
	return Value{typ: t}
}

// copyValue copies a stored value: arrays and structs copy deeply
// (fresh backing), slices share their backing store, maps share.
func copyValue(v Value) Value {
	switch {
	case v.array != nil:
		out := make([]Value, len(v.array))
		for i, e := range v.array {
			out[i] = copyValue(e)
		}
		v.array = out
	case v.strct != nil:
		out := make([]Value, len(v.strct))
		for i, e := range v.strct {
			out[i] = copyValue(e)
		}
		v.strct = out
	case v.slice != nil:
		h := *v.slice
		v.slice = &h
	}
	return v
}

// copyArg returns a deep copy of a call argument. A const func writes
// through its parameters, and a map or slice argument can carry a recorded
// const's value, so the callee must not reach the caller's storage (SPEC
// T-201, T-Sec9). Arrays, structs, slice backings, and map entries copy
// recursively.
func copyArg(v Value) Value {
	switch {
	case v.array != nil:
		out := make([]Value, len(v.array))
		for i, e := range v.array {
			out[i] = copyArg(e)
		}
		v.array = out
	case v.strct != nil:
		out := make([]Value, len(v.strct))
		for i, e := range v.strct {
			out[i] = copyArg(e)
		}
		v.strct = out
	case v.slice != nil:
		src := *v.slice.backing
		backing := make([]Value, len(src))
		for i, e := range src {
			backing[i] = copyArg(e)
		}
		h := *v.slice
		h.backing = &backing
		v.slice = &h
	case v.mval != nil:
		m := &mapVal{keys: make([]Value, len(v.mval.keys)), vals: make([]Value, len(v.mval.vals))}
		for i := range v.mval.keys {
			m.keys[i] = copyArg(v.mval.keys[i])
			m.vals[i] = copyArg(v.mval.vals[i])
		}
		v.mval = m
	}
	return v
}

// resolveType prechecks purity-class type shapes, then delegates to
// the host. Named types resolving behind pointers, channels, or
// uintptr arrive with Impure set and report the same reason.
func (e *evaluator) resolveType(fr *frame, te syntax.Expr) (*Type, error) {
	if err := e.precheckType(fr, te); err != nil {
		return nil, err
	}
	t := e.h.TypeOf(te)
	if t == nil {
		return nil, e.fail(posOf(te), "unsupported type in const evaluation")
	}
	if t.Impure != "" {
		return nil, e.impurity(fr, posOf(te), "%s", t.Impure)
	}
	return t, nil
}

// precheckType flags pointer, channel, and uintptr shapes before
// resolution, recursing through composite shapes and const views.
func (e *evaluator) precheckType(fr *frame, te syntax.Expr) error {
	switch te := syntax.Unparen(te).(type) {
	case *syntax.Operation:
		if te.Op == syntax.Mul && te.Y == nil {
			return e.impurity(fr, posOf(te), "pointer type in const func")
		}
	case *syntax.ChanType:
		return e.impurity(fr, posOf(te), "channel type in const func")
	case *syntax.FuncType:
		return e.fail(posOf(te), "function types in const func not supported")
	case *syntax.InterfaceType:
		return e.fail(posOf(te), "interface types in const func not supported")
	case *syntax.Name:
		if te.Value == "uintptr" {
			return e.impurity(fr, posOf(te), "uintptr type in const func")
		}
		if te.Value == "any" || te.Value == "error" {
			return e.fail(posOf(te), "interface types in const func not supported")
		}
	case *syntax.SelectorExpr:
		if x, ok := te.X.(*syntax.Name); ok && x.Value == "unsafe" && te.Sel.Value == "Pointer" {
			return e.impurity(fr, posOf(te), "pointer type in const func")
		}
	case *syntax.SliceType:
		return e.precheckType(fr, te.Elem)
	case *syntax.ArrayType:
		return e.precheckType(fr, te.Elem)
	case *syntax.MapType:
		if err := e.precheckType(fr, te.Key); err != nil {
			return err
		}
		return e.precheckType(fr, te.Value)
	case *syntax.StructType:
		for _, f := range te.FieldList {
			if err := e.precheckType(fr, f.Type); err != nil {
				return err
			}
		}
	case *syntax.DotsType:
		return e.precheckType(fr, te.Elem)
	case *syntax.ConstType:
		return e.precheckType(fr, te.Elem)
	case *syntax.OptionType:
		return e.fail(posOf(te), "option types in const func not supported")
	case *syntax.IndexExpr:
		return e.fail(posOf(te), "generic types in const func not supported")
	}
	return nil
}

// evalConversion evaluates T(x) and string/bytes/rune conversions.
func (e *evaluator) evalConversion(fr *frame, c *syntax.CallExpr) (Value, error) {
	if err := e.precheckType(fr, c.Fun); err != nil {
		return Value{}, err
	}
	t := e.h.TypeOf(c)
	if t == nil {
		return Value{}, e.fail(posOf(c), "unsupported conversion in const evaluation")
	}
	if t.Impure != "" {
		return Value{}, e.impurity(fr, posOf(c), "%s", t.Impure)
	}
	if len(c.ArgList) != 1 || c.HasDots {
		return Value{}, e.fail(posOf(c), "bad conversion in const evaluation")
	}
	v, err := e.evalExpr(fr, c.ArgList[0])
	if err != nil {
		return Value{}, err
	}
	return e.convert(v, t, posOf(c))
}

// convert implements Go conversion semantics with target widths:
// constant sources range-check, variable sources wrap or truncate.
func (e *evaluator) convert(v Value, t *Type, pos syntax.Pos) (Value, error) {
	if t.Kind == KUintptr {
		return Value{}, e.impurity(nil, pos, "conversion to uintptr in const func")
	}
	if v.isNil {
		return e.coerceNil(v, t, pos)
	}
	vk := kindOf(v)
	if vk == t.Kind {
		v.typ = t
		return v, nil
	}
	// Named types share their underlying kind; kinds match above.
	if v.typ != nil && v.typ.Kind == t.Kind {
		v.typ = t
		return v, nil
	}
	switch t.Kind {
	case KString:
		return e.convertToString(v, t, pos)
	case KSlice:
		if vk == KString && t.Elem != nil && (t.Elem.Kind == KUint8 || t.Elem.Kind == KInt32) {
			return e.stringToSlice(v, t, pos)
		}
	}
	if !isUntyped(vk) && vk == KString && t.Kind == KSlice {
		return Value{}, e.fail(pos, "cannot convert string in const evaluation")
	}
	if isInteger(t.Kind) || isFloat(t.Kind) {
		return e.convertNumber(v, t, pos)
	}
	if t.Kind == KBool && (vk == KBool || vk == KUntypedBool) {
		v.typ = t
		return v, nil
	}
	return Value{}, e.fail(pos, "cannot convert %s in const evaluation", typeName(v.typ))
}

// coerceNil adopts a target type for nil: slices and maps go nil.
func (e *evaluator) coerceNil(v Value, t *Type, pos syntax.Pos) (Value, error) {
	switch t.Kind {
	case KSlice, KMap:
		return Value{typ: t, isNil: true}, nil
	}
	return Value{}, e.fail(pos, "cannot use nil as %s in const evaluation", typeName(t))
}

// convertNumber converts between numeric kinds.
func (e *evaluator) convertNumber(v Value, t *Type, pos syntax.Pos) (Value, error) {
	vk := kindOf(v)
	if !isInteger(vk) && !isFloat(vk) {
		return Value{}, e.fail(pos, "cannot convert %s to %s in const evaluation", typeName(v.typ), typeName(t))
	}
	if isInteger(t.Kind) {
		var n *big.Int
		if isFloat(vk) {
			f := xFloat(v)
			if v.constant && !f.IsInt() {
				return Value{}, e.fail(pos, "constant truncated to integer in const evaluation")
			}
			n = new(big.Int)
			if v.constant {
				n.Div(f.Num(), f.Denom())
			} else {
				f64 := toFloat64(v, v.typ)
				if math.IsNaN(f64) || math.IsInf(f64, 0) {
					return Value{}, e.fail(pos, "invalid float to integer conversion in const evaluation")
				}
				n.SetInt64(int64(f64))
			}
		} else {
			n = new(big.Int).Set(v.ival)
		}
		if t.Kind == KUntypedInt || t.Kind == KUntypedRune {
			return intValue(n, t, v.constant), nil
		}
		if v.constant {
			if !inRange(n, t, e.cfg.WordSize) {
				return Value{}, e.fail(pos, "constant %v overflows %s in const evaluation", n, typeName(t))
			}
			return intValue(n, t, true), nil
		}
		return e.wrapInt(n, t, false, pos)
	}
	// Float target.
	var f float64
	if isFloat(vk) {
		if v.constant {
			r := xFloat(v)
			f, _ = r.Float64()
		} else {
			f = toFloat64(v, v.typ)
		}
	} else {
		r := new(big.Rat).SetInt(v.ival)
		f, _ = r.Float64()
	}
	if v.constant {
		if t.Kind == KFloat32 && math.IsInf(float64(float32(f)), 0) && !math.IsInf(f, 0) {
			return Value{}, e.fail(pos, "constant overflows float32 in const evaluation")
		}
		if t.Kind == KFloat64 && math.IsInf(f, 0) {
			// Overflow through conversion still fits Inf? Constants
			// must be representable: Inf is not.
			if !isFloat(vk) || !v.ival.IsInt64() {
				return Value{}, e.fail(pos, "constant overflows float64 in const evaluation")
			}
		}
		if isUntyped(t.Kind) {
			return Value{typ: t, constant: true, fval: new(big.Rat).SetFloat64(f)}, nil
		}
		return floatResult(t, f, true, pos)
	}
	return floatResult(t, f, false, pos)
}

// convertToString implements string(x) for integers (rune), floats are
// rejected, and strings pass through.
func (e *evaluator) convertToString(v Value, t *Type, pos syntax.Pos) (Value, error) {
	vk := kindOf(v)
	if vk == KString || vk == KUntypedString {
		return Value{typ: t, constant: v.constant, sval: v.sval}, nil
	}
	if !isInteger(vk) {
		return Value{}, e.fail(pos, "cannot convert %s to string in const evaluation", typeName(v.typ))
	}
	n := v.ival.Int64()
	if v.constant {
		if n < 0 || n > utf8.MaxRune || (n >= 0xD800 && n <= 0xDFFF) {
			return Value{}, e.fail(pos, "constant %v is not a valid rune in const evaluation", v.ival)
		}
		return Value{typ: t, constant: true, sval: string(rune(n))}, nil
	}
	r := rune(n)
	if n < 0 || n > utf8.MaxRune || (n >= 0xD800 && n <= 0xDFFF) {
		return Value{typ: t, sval: "\uFFFD"}, nil
	}
	return Value{typ: t, sval: string(r)}, nil
}

// stringToSlice implements []byte(s) and []rune(s).
func (e *evaluator) stringToSlice(v Value, t *Type, pos syntax.Pos) (Value, error) {
	var elems []Value
	if t.Elem.Kind == KUint8 {
		for _, b := range []byte(v.sval) {
			elems = append(elems, byteValue(b))
		}
	} else {
		for _, r := range v.sval {
			elems = append(elems, runeValue(r))
		}
	}
	return sliceValue(t.Elem, elems, len(elems)), nil
}

// evalComposite evaluates struct, array, slice, and map literals.
// expected supplies the element type for elided nested literals.
func (e *evaluator) evalComposite(fr *frame, lit *syntax.CompositeLit, expected *Type) (Value, error) {
	var t *Type
	if lit.Type != nil {
		var err error
		t, err = e.resolveType(fr, lit.Type)
		if err != nil {
			return Value{}, err
		}
		if t.Kind == KArray && t.Len < 0 {
			t = cloneType(t)
			t.Len = int64(len(lit.ElemList))
		}
	} else if expected != nil {
		t = expected
	} else {
		return Value{}, e.fail(posOf(lit), "missing literal type in const evaluation")
	}
	switch t.Kind {
	case KSlice:
		return e.evalSliceLit(fr, lit, t)
	case KArray:
		return e.evalArrayLit(fr, lit, t)
	case KStruct:
		return e.evalStructLit(fr, lit, t)
	case KMap:
		return e.evalMapLit(fr, lit, t)
	}
	return Value{}, e.fail(posOf(lit), "composite of %s not supported", typeName(t))
}

func cloneType(t *Type) *Type {
	c := *t
	return &c
}

func (e *evaluator) evalSliceLit(fr *frame, lit *syntax.CompositeLit, t *Type) (Value, error) {
	elems, err := e.evalElements(fr, lit, t.Elem)
	if err != nil {
		return Value{}, err
	}
	v := sliceValue(t.Elem, elems, len(elems))
	v.typ = t
	return v, nil
}

func (e *evaluator) evalArrayLit(fr *frame, lit *syntax.CompositeLit, t *Type) (Value, error) {
	n := int(t.Len)
	elems := make([]Value, n)
	for i := range elems {
		elems[i] = zeroValue(t.Elem)
	}
	pos := posOf(lit)
	next := 0
	for _, el := range lit.ElemList {
		if kv, ok := el.(*syntax.KeyValueExpr); ok {
			k, err := e.evalExpr(fr, kv.Key)
			if err != nil {
				return Value{}, err
			}
			if !isInteger(kindOf(k)) {
				return Value{}, e.fail(posOf(kv.Key), "non-integer index in const evaluation")
			}
			idx := int(k.ival.Int64())
			if idx < 0 || idx >= n {
				return Value{}, e.fail(posOf(kv.Key), "index out of range in const evaluation")
			}
			v, err := e.evalElem(fr, kv.Value, t.Elem)
			if err != nil {
				return Value{}, err
			}
			elems[idx] = v
			next = idx + 1
			continue
		}
		if next >= n {
			return Value{}, e.fail(pos, "too many initializers in const evaluation")
		}
		v, err := e.evalElem(fr, el, t.Elem)
		if err != nil {
			return Value{}, err
		}
		elems[next] = v
		next++
	}
	for i := range elems {
		elems[i] = copyValue(withVar(elems[i]))
	}
	return Value{typ: t, array: elems}, nil
}

// evalElements evaluates slice literal elements with optional indices.
func (e *evaluator) evalElements(fr *frame, lit *syntax.CompositeLit, elemT *Type) ([]Value, error) {
	var elems []Value
	next := 0
	for _, el := range lit.ElemList {
		if kv, ok := el.(*syntax.KeyValueExpr); ok {
			k, err := e.evalExpr(fr, kv.Key)
			if err != nil {
				return nil, err
			}
			if !isInteger(kindOf(k)) {
				return nil, e.fail(posOf(kv.Key), "non-integer index in const evaluation")
			}
			idx := int(k.ival.Int64())
			if idx < 0 {
				return nil, e.fail(posOf(kv.Key), "index out of range in const evaluation")
			}
			for len(elems) <= idx {
				elems = append(elems, zeroValue(elemT))
			}
			v, err := e.evalElem(fr, kv.Value, elemT)
			if err != nil {
				return nil, err
			}
			elems[idx] = v
			next = idx + 1
			continue
		}
		v, err := e.evalElem(fr, el, elemT)
		if err != nil {
			return nil, err
		}
		elems = append(elems, v)
		next++
	}
	_ = next
	for i := range elems {
		elems[i] = copyValue(withVar(elems[i]))
	}
	return elems, nil
}

// evalElem evaluates one composite element against its static type,
// descending into elided nested literals.
func (e *evaluator) evalElem(fr *frame, el syntax.Expr, elemT *Type) (Value, error) {
	if inner, ok := syntax.Unparen(el).(*syntax.CompositeLit); ok && inner.Type == nil {
		v, err := e.evalComposite(fr, inner, elemT)
		if err != nil {
			return Value{}, err
		}
		return v, nil
	}
	v, err := e.evalExpr(fr, el)
	if err != nil {
		return Value{}, err
	}
	if elemT != nil {
		if v.isNil {
			return e.coerceNil(v, elemT, posOf(el))
		}
		if v, err = e.convert(v, elemT, posOf(el)); err != nil {
			return Value{}, err
		}
	}
	return v, nil
}

func (e *evaluator) evalStructLit(fr *frame, lit *syntax.CompositeLit, t *Type) (Value, error) {
	fields := make([]Value, len(t.Fields))
	for i, f := range t.Fields {
		fields[i] = zeroValue(f.Type)
	}
	index := map[string]int{}
	for i, f := range t.Fields {
		index[f.Name] = i
	}
	pos := posOf(lit)
	next := 0
	keyed := false
	for _, el := range lit.ElemList {
		if kv, ok := el.(*syntax.KeyValueExpr); ok {
			keyed = true
			name, ok := syntax.Unparen(kv.Key).(*syntax.Name)
			if !ok {
				return Value{}, e.fail(posOf(kv.Key), "bad field key in const evaluation")
			}
			i, ok := index[name.Value]
			if !ok {
				return Value{}, e.fail(posOf(kv.Key), "unknown field in const evaluation")
			}
			v, err := e.evalElem(fr, kv.Value, t.Fields[i].Type)
			if err != nil {
				return Value{}, err
			}
			fields[i] = copyValue(withVar(v))
			continue
		}
		if keyed {
			return Value{}, e.fail(pos, "mixed keyed and unkeyed fields in const evaluation")
		}
		if next >= len(fields) {
			return Value{}, e.fail(pos, "too many initializers in const evaluation")
		}
		v, err := e.evalElem(fr, el, t.Fields[next].Type)
		if err != nil {
			return Value{}, err
		}
		fields[next] = copyValue(withVar(v))
		next++
	}
	return Value{typ: t, strct: fields}, nil
}

func (e *evaluator) evalMapLit(fr *frame, lit *syntax.CompositeLit, t *Type) (Value, error) {
	mv := &mapVal{}
	v := Value{typ: t, mval: mv}
	for _, el := range lit.ElemList {
		kv, ok := el.(*syntax.KeyValueExpr)
		if !ok {
			return Value{}, e.fail(posOf(el), "missing key in const map")
		}
		k, err := e.evalExpr(fr, kv.Key)
		if err != nil {
			return Value{}, err
		}
		if t.Key != nil {
			if k, err = e.convert(k, t.Key, posOf(kv.Key)); err != nil {
				return Value{}, err
			}
		}
		val, err := e.evalElem(fr, kv.Value, t.Elem)
		if err != nil {
			return Value{}, err
		}
		if err := e.mapInsert(v, k, copyValue(withVar(val)), posOf(el)); err != nil {
			return Value{}, err
		}
	}
	return v, nil
}

// mapInsert inserts with -0.0 canonicalization and NaN rejection.
func (e *evaluator) mapInsert(m, k, v Value, pos syntax.Pos) error {
	if m.isNil {
		return e.fail(pos, "assignment to nil map in const evaluation")
	}
	if isNaNKey(k) {
		return &Error{Code: CodeEval, Detail: "NaN is not a valid const map key", Pos: pos}
	}
	k = e.canonKey(k)
	for i, ek := range m.mval.keys {
		eq, err := e.equal(ek, k, pos)
		if err != nil {
			return err
		}
		if eq {
			return &Error{Code: CodeEval, Detail: "duplicate key in const map", Pos: pos}
		}
		_ = i
	}
	m.mval.keys = append(m.mval.keys, k)
	m.mval.vals = append(m.mval.vals, v)
	return nil
}

func isNaNKey(k Value) bool {
	if k.typ == nil {
		return false
	}
	if k.typ.Kind == KFloat32 || k.typ.Kind == KFloat64 {
		return math.IsNaN(k.Float64())
	}
	return false
}

// canonKey canonicalizes map keys: -0.0 becomes +0.0, recursively
// through structs and arrays (SPEC T-Sec9).
func (e *evaluator) canonKey(v Value) Value {
	if v.typ != nil && (v.typ.Kind == KFloat32 || v.typ.Kind == KFloat64) && !v.constant {
		if v.f64 == 0 {
			v.f64 = 0
		}
	}
	if v.typ != nil && v.typ.Kind == KStruct {
		out := slices.Clone(v.strct)
		for i := range out {
			out[i] = e.canonKey(out[i])
		}
		v.strct = out
	}
	if v.typ != nil && v.typ.Kind == KArray {
		out := slices.Clone(v.array)
		for i := range out {
			out[i] = e.canonKey(out[i])
		}
		v.array = out
	}
	return v
}

// evalSelector evaluates struct field access.
func (e *evaluator) evalSelector(fr *frame, sel *syntax.SelectorExpr) (Value, error) {
	if name, ok := sel.X.(*syntax.Name); ok {
		if ent := e.h.Resolve(name); ent.Kind == EntPackage {
			rent := e.h.ResolveSel(sel)
			switch rent.Kind {
			case EntConst:
				return rent.Const, nil
			case EntPackageVar:
				return Value{}, e.impurity(fr, posOf(sel), "read of package variable %s in const func", rent.Name)
			}
			return Value{}, e.impurity(fr, posOf(sel), "use of imported %s in const func", rent.Name)
		}
	}
	x, err := e.evalExpr(fr, sel.X)
	if err != nil {
		return Value{}, err
	}
	if x.typ == nil || x.typ.Kind != KStruct {
		return Value{}, e.impurity(fr, posOf(sel), "method call in const func")
	}
	for i, f := range x.typ.Fields {
		if f.Name == sel.Sel.Value {
			return x.strct[i], nil
		}
	}
	return Value{}, e.impurity(fr, posOf(sel), "method call in const func")
}

// fieldPlace resolves a struct field store.
func (e *evaluator) fieldPlace(fr *frame, sel *syntax.SelectorExpr) (place, error) {
	if name, ok := sel.X.(*syntax.Name); ok {
		if ent := e.h.Resolve(name); ent.Kind == EntPackage {
			return place{}, e.impurity(fr, posOf(sel), "write to imported %s in const func", sel.Sel.Value)
		}
	}
	xv, err := e.evalExpr(fr, sel.X)
	if err != nil {
		return place{}, err
	}
	if xv.typ == nil || xv.typ.Kind != KStruct {
		return place{}, e.impurity(fr, posOf(sel), "method call in const func")
	}
	// Struct values are addressable here; find the base place when the
	// receiver is a plain variable so stores persist.
	if base, ok := syntax.Unparen(sel.X).(*syntax.Name); ok {
		for i := len(fr.scopes) - 1; i >= 0; i-- {
			if _, ok := fr.scopes[i][base.Value]; ok {
				idx := i
				name := base.Value
				for fi, f := range xv.typ.Fields {
					if f.Name == sel.Sel.Value {
						fidx := fi
						return place{
							get: func() Value { return fr.scopes[idx][name].strct[fidx] },
							set: func(v Value) {
								cur := fr.scopes[idx][name]
								cur.strct[fidx] = copyValue(v)
								fr.scopes[idx][name] = cur
							},
						}, nil
					}
				}
			}
		}
	}
	return place{}, e.fail(posOf(sel), "cannot assign struct field in const evaluation")
}

// evalIndex evaluates index expressions.
func (e *evaluator) evalIndex(fr *frame, ix *syntax.IndexExpr) (Value, error) {
	if _, ok := ix.Index.(*syntax.ListExpr); ok {
		return Value{}, e.fail(posOf(ix), "generic instantiation in const func not supported")
	}
	x, err := e.evalExpr(fr, ix.X)
	if err != nil {
		return Value{}, err
	}
	k, err := e.evalExpr(fr, ix.Index)
	if err != nil {
		return Value{}, err
	}
	return e.indexValue(x, k, posOf(ix))
}

func (e *evaluator) indexValue(x, k Value, pos syntax.Pos) (Value, error) {
	switch {
	case x.slice != nil:
		if !isInteger(kindOf(k)) {
			return Value{}, e.fail(pos, "non-integer index in const evaluation")
		}
		i := int(k.ival.Int64())
		elems := sliceElems(x)
		if i < 0 || i >= len(elems) {
			return Value{}, e.fail(pos, "index out of range in const evaluation")
		}
		return withVar(elems[i]), nil
	case x.array != nil:
		if !isInteger(kindOf(k)) {
			return Value{}, e.fail(pos, "non-integer index in const evaluation")
		}
		i := int(k.ival.Int64())
		if i < 0 || i >= len(x.array) {
			return Value{}, e.fail(pos, "index out of range in const evaluation")
		}
		return withVar(x.array[i]), nil
	case x.typ != nil && x.typ.Kind == KString:
		if !isInteger(kindOf(k)) {
			return Value{}, e.fail(pos, "non-integer index in const evaluation")
		}
		i := int(k.ival.Int64())
		if i < 0 || i >= len(x.sval) {
			return Value{}, e.fail(pos, "index out of range in const evaluation")
		}
		return byteValue(x.sval[i]), nil
	case x.mval != nil:
		v, present, err := e.mapLookup(x, k, pos)
		if err != nil {
			return Value{}, err
		}
		if !present {
			return Value{}, e.fail(pos, "key not present in const map")
		}
		return withVar(v), nil
	}
	return Value{}, e.fail(pos, "cannot index in const evaluation")
}

// mapLookup finds a canonicalized key, reporting presence.
func (e *evaluator) mapLookup(x, k Value, pos syntax.Pos) (Value, bool, error) {
	if isNaNKey(k) {
		return Value{}, false, &Error{Code: CodeEval, Detail: "NaN is not a valid const map key", Pos: pos}
	}
	k = e.canonKey(k)
	for i, ek := range x.mval.keys {
		eq, err := e.equal(ek, k, pos)
		if err != nil {
			return Value{}, false, err
		}
		if eq {
			return x.mval.vals[i], true, nil
		}
	}
	return Value{}, false, nil
}

// indexPlace resolves an index store.
func (e *evaluator) indexPlace(fr *frame, ix *syntax.IndexExpr) (place, error) {
	if _, ok := ix.Index.(*syntax.ListExpr); ok {
		return place{}, e.fail(posOf(ix), "generic instantiation in const func not supported")
	}
	k, err := e.evalExpr(fr, ix.Index)
	if err != nil {
		return place{}, err
	}
	// Map stores create missing keys.
	if base, ok := syntax.Unparen(ix.X).(*syntax.Name); ok {
		if bv, found := e.lookup(fr, base.Value); found && bv.mval != nil {
			for i := len(fr.scopes) - 1; i >= 0; i-- {
				if _, ok := fr.scopes[i][base.Value]; ok {
					idx := i
					name := base.Value
					return place{
						get: func() Value {
							v, err := e.indexValue(fr.scopes[idx][name], k, posOf(ix))
							if err != nil {
								return zeroValue(&Type{Kind: KInt})
							}
							return v
						},
						set: func(v Value) {
							cur := fr.scopes[idx][name]
							if isNaNKey(k) {
								return
							}
							kk := e.canonKey(k)
							for i, ek := range cur.mval.keys {
								if eq, _ := e.equal(ek, kk, posOf(ix)); eq {
									cur.mval.vals[i] = copyValue(v)
									return
								}
							}
							cur.mval.keys = append(cur.mval.keys, kk)
							cur.mval.vals = append(cur.mval.vals, copyValue(v))
						},
					}, nil
				}
			}
		}
	}
	x, err := e.evalExpr(fr, ix.X)
	if err != nil {
		return place{}, err
	}
	if x.slice != nil {
		if !isInteger(kindOf(k)) {
			return place{}, e.fail(posOf(ix), "non-integer index in const evaluation")
		}
		i := int(k.ival.Int64())
		elems := sliceElems(x)
		if i < 0 || i >= len(elems) {
			return place{}, e.fail(posOf(ix), "index out of range in const evaluation")
		}
		// Resolve through the backing store so aliases observe it.
		backing := x.slice.backing
		at := x.slice.lo + i
		return place{
			get: func() Value { return (*backing)[at] },
			set: func(v Value) { (*backing)[at] = copyValue(v) },
		}, nil
	}
	if x.array != nil {
		if _, ok := syntax.Unparen(ix.X).(*syntax.Name); !ok {
			return place{}, e.fail(posOf(ix), "cannot assign array element in const evaluation")
		}
		// Arrays copy on assignment; element stores persist only
		// through variables, resolved below.
		if !isInteger(kindOf(k)) {
			return place{}, e.fail(posOf(ix), "non-integer index in const evaluation")
		}
		i := int(k.ival.Int64())
		if i < 0 || i >= len(x.array) {
			return place{}, e.fail(posOf(ix), "index out of range in const evaluation")
		}
		if base, ok := syntax.Unparen(ix.X).(*syntax.Name); ok {
			for si := len(fr.scopes) - 1; si >= 0; si-- {
				if _, ok := fr.scopes[si][base.Value]; ok {
					sidx := si
					name := base.Value
					return place{
						get: func() Value { return fr.scopes[sidx][name].array[i] },
						set: func(v Value) {
							cur := fr.scopes[sidx][name]
							cur.array[i] = copyValue(v)
							fr.scopes[sidx][name] = cur
						},
					}, nil
				}
			}
		}
	}
	return place{}, e.fail(posOf(ix), "cannot assign in const evaluation")
}

// evalSlice evaluates two- and three-index slice expressions.
func (e *evaluator) evalSlice(fr *frame, sx *syntax.SliceExpr) (Value, error) {
	x, err := e.evalExpr(fr, sx.X)
	if err != nil {
		return Value{}, err
	}
	bound := func(bx syntax.Expr, def int64) (int64, error) {
		if bx == nil {
			return def, nil
		}
		v, err := e.evalExpr(fr, bx)
		if err != nil {
			return 0, err
		}
		if !isInteger(kindOf(v)) {
			return 0, e.fail(posOf(bx), "non-integer slice bound in const evaluation")
		}
		return v.ival.Int64(), nil
	}
	switch {
	case x.slice != nil:
		n := int64(len(sliceElems(x)))
		c := int64(x.slice.cap_ - x.slice.lo)
		lo, err := bound(sx.Index[0], 0)
		if err != nil {
			return Value{}, err
		}
		hi, err := bound(sx.Index[1], n)
		if err != nil {
			return Value{}, err
		}
		max, err := bound(sx.Index[2], c)
		if err != nil {
			return Value{}, err
		}
		if lo < 0 || hi < lo || hi > n || max < hi || max > c {
			return Value{}, e.fail(posOf(sx), "slice bounds out of range in const evaluation")
		}
		nv := x
		nh := *x.slice
		nh.lo += int(lo)
		nh.length = int(hi - lo)
		nh.cap_ = int(max)
		nv.slice = &nh
		return nv, nil
	case x.array != nil:
		n := int64(len(x.array))
		lo, err := bound(sx.Index[0], 0)
		if err != nil {
			return Value{}, err
		}
		hi, err := bound(sx.Index[1], n)
		if err != nil {
			return Value{}, err
		}
		if sx.Index[2] != nil {
			return Value{}, e.fail(posOf(sx), "3-index slice of array in const evaluation")
		}
		if lo < 0 || hi < lo || hi > n {
			return Value{}, e.fail(posOf(sx), "slice bounds out of range in const evaluation")
		}
		// Slicing an array copies into a fresh backing.
		elems := slices.Clone(x.array[lo:hi])
		return sliceValue(x.typ.Elem, elems, int(hi-lo)), nil
	case x.typ != nil && x.typ.Kind == KString:
		n := int64(len(x.sval))
		lo, err := bound(sx.Index[0], 0)
		if err != nil {
			return Value{}, err
		}
		hi, err := bound(sx.Index[1], n)
		if err != nil {
			return Value{}, err
		}
		if sx.Index[2] != nil {
			return Value{}, e.fail(posOf(sx), "3-index slice of string in const evaluation")
		}
		if lo < 0 || hi < lo || hi > n {
			return Value{}, e.fail(posOf(sx), "slice bounds out of range in const evaluation")
		}
		return Value{typ: x.typ, constant: false, sval: x.sval[lo:hi]}, nil
	}
	return Value{}, e.fail(posOf(sx), "cannot slice in const evaluation")
}

// evalBuiltin evaluates builtin calls.
func (e *evaluator) evalBuiltin(fr *frame, name string, c *syntax.CallExpr) (Value, error) {
	pos := posOf(c)
	switch name {
	case "len", "cap":
		if len(c.ArgList) != 1 || c.HasDots {
			return Value{}, e.fail(pos, "bad %s call in const evaluation", name)
		}
		v, err := e.evalExpr(fr, c.ArgList[0])
		if err != nil {
			return Value{}, err
		}
		switch {
		case v.typ != nil && v.typ.Kind == KString:
			return intValue(big.NewInt(int64(len(v.sval))), UntypedInt(), true), nil
		case v.slice != nil:
			if name == "len" {
				return intValue(big.NewInt(int64(len(sliceElems(v)))), UntypedInt(), true), nil
			}
			return intValue(big.NewInt(int64(v.slice.cap_-v.slice.lo)), UntypedInt(), true), nil
		case v.array != nil:
			return intValue(big.NewInt(int64(len(v.array))), UntypedInt(), true), nil
		case v.mval != nil:
			if name == "len" {
				return intValue(big.NewInt(int64(len(v.mval.keys))), UntypedInt(), true), nil
			}
		}
		return Value{}, e.fail(pos, "bad %s call in const evaluation", name)
	case "append":
		return e.evalAppend(fr, c)
	case "copy":
		return e.evalCopy(fr, c)
	case "make":
		return e.evalMake(fr, c)
	case "delete":
		if len(c.ArgList) != 2 || c.HasDots {
			return Value{}, e.fail(pos, "bad delete call in const evaluation")
		}
		m, err := e.evalExpr(fr, c.ArgList[0])
		if err != nil {
			return Value{}, err
		}
		k, err := e.evalExpr(fr, c.ArgList[1])
		if err != nil {
			return Value{}, err
		}
		if m.mval == nil {
			return Value{}, e.fail(pos, "delete of non-map in const evaluation")
		}
		if m.isNil {
			return zeroValue(&Type{Kind: KInt}), nil
		}
		if isNaNKey(k) {
			return Value{}, &Error{Code: CodeEval, Detail: "NaN is not a valid const map key", Pos: pos}
		}
		k = e.canonKey(k)
		for i, ek := range m.mval.keys {
			if eq, _ := e.equal(ek, k, pos); eq {
				m.mval.keys = slices.Delete(m.mval.keys, i, i+1)
				m.mval.vals = slices.Delete(m.mval.vals, i, i+1)
				break
			}
		}
		return zeroValue(&Type{Kind: KInt}), nil
	case "clear":
		if len(c.ArgList) != 1 || c.HasDots {
			return Value{}, e.fail(pos, "bad clear call in const evaluation")
		}
		v, err := e.evalExpr(fr, c.ArgList[0])
		if err != nil {
			return Value{}, err
		}
		switch {
		case v.mval != nil:
			if !v.isNil {
				v.mval.keys = nil
				v.mval.vals = nil
			}
			return zeroValue(&Type{Kind: KInt}), nil
		case v.slice != nil:
			zero := zeroValue(v.typ.Elem)
			for i := range sliceLen(v) {
				(*v.slice.backing)[v.slice.lo+i] = zero
			}
			return zeroValue(&Type{Kind: KInt}), nil
		}
		return Value{}, e.fail(pos, "bad clear call in const evaluation")
	case "min":
		return e.evalExtremum(syntax.Lss, fr, c)
	case "max":
		return e.evalExtremum(syntax.Gtr, fr, c)
	case "panic":
		if len(c.ArgList) != 1 {
			return Value{}, e.fail(pos, "bad panic call in const evaluation")
		}
		v, err := e.evalExpr(fr, c.ArgList[0])
		if err != nil {
			return Value{}, err
		}
		return Value{}, &Error{Code: CodeEval, Detail: "panic in const func: " + e.summarize(v), Pos: pos}
	case "print", "println":
		return Value{}, e.impurity(fr, pos, "I/O: call to %s in const func", name)
	case "recover":
		return Value{}, e.impurity(fr, pos, "call to recover in const func")
	case "new":
		return Value{}, e.impurity(fr, pos, "pointer allocation in const func")
	case "real", "imag", "complex":
		return Value{}, e.fail(pos, "complex numbers in const func not supported")
	}
	return Value{}, e.fail(pos, "unsupported builtin %s in const evaluation", name)
}

func (e *evaluator) summarize(v Value) string {
	switch {
	case v.typ != nil && isInteger(v.typ.Kind) && v.ival != nil:
		return v.ival.String()
	case v.typ != nil && v.typ.Kind == KString:
		return strconv.Quote(v.sval)
	case v.typ != nil && (v.typ.Kind == KBool || v.typ.Kind == KUntypedBool):
		return strconv.FormatBool(v.bval)
	}
	return typeName(v.typ)
}

// evalAppend implements append with Go growth and sharing: elements
// extend shared backing while capacity allows.
func (e *evaluator) evalAppend(fr *frame, c *syntax.CallExpr) (Value, error) {
	pos := posOf(c)
	if len(c.ArgList) < 1 {
		return Value{}, e.fail(pos, "bad append call in const evaluation")
	}
	s, err := e.evalExpr(fr, c.ArgList[0])
	if err != nil {
		return Value{}, err
	}
	if s.slice == nil {
		return Value{}, e.fail(pos, "append to non-slice in const evaluation")
	}
	elems, err := e.evalArgs(fr, &syntax.CallExpr{ArgList: c.ArgList[1:], HasDots: c.HasDots})
	if err != nil {
		return Value{}, err
	}
	for i := range elems {
		elems[i] = copyValue(withVar(elems[i]))
		if s.typ.Elem != nil {
			if elems[i], err = e.convert(elems[i], s.typ.Elem, pos); err != nil {
				return Value{}, err
			}
		}
	}
	lo, length, cap_ := s.slice.lo, s.slice.length, s.slice.cap_
	if length+len(elems) <= cap_-lo {
		copy((*s.slice.backing)[lo+length:], elems)
		nh := *s.slice
		nh.length += len(elems)
		s.slice = &nh
		return s, nil
	}
	need := length + len(elems)
	newcap := 2 * (cap_ - lo)
	if newcap < need {
		newcap = need
	}
	if newcap == 0 {
		return s, nil
	}
	backing := make([]Value, newcap)
	copy(backing, sliceElems(s))
	copy(backing[length:], elems)
	s.slice = &sliceVal{backing: &backing, length: need, cap_: newcap}
	return s, nil
}

// evalCopy implements copy(dst, src).
func (e *evaluator) evalCopy(fr *frame, c *syntax.CallExpr) (Value, error) {
	pos := posOf(c)
	if len(c.ArgList) != 2 || c.HasDots {
		return Value{}, e.fail(pos, "bad copy call in const evaluation")
	}
	dst, err := e.evalExpr(fr, c.ArgList[0])
	if err != nil {
		return Value{}, err
	}
	src, err := e.evalExpr(fr, c.ArgList[1])
	if err != nil {
		return Value{}, err
	}
	if dst.slice == nil {
		return Value{}, e.fail(pos, "copy to non-slice in const evaluation")
	}
	var selems []Value
	if src.slice != nil {
		selems = sliceElems(src)
	} else if src.typ != nil && src.typ.Kind == KString {
		for _, b := range []byte(src.sval) {
			selems = append(selems, byteValue(b))
		}
	} else {
		return Value{}, e.fail(pos, "bad copy call in const evaluation")
	}
	delems := sliceElems(dst)
	n := min(len(delems), len(selems))
	for i := range n {
		v := copyValue(selems[i])
		if dst.typ.Elem != nil {
			var err error
			if v, err = e.convert(v, dst.typ.Elem, pos); err != nil {
				return Value{}, err
			}
		}
		(*dst.slice.backing)[dst.slice.lo+i] = v
	}
	return intValue(big.NewInt(int64(n)), UntypedInt(), true), nil
}

// evalMake implements make for slices and maps.
func (e *evaluator) evalMake(fr *frame, c *syntax.CallExpr) (Value, error) {
	pos := posOf(c)
	if len(c.ArgList) < 1 {
		return Value{}, e.fail(pos, "bad make call in const evaluation")
	}
	if err := e.precheckType(fr, c.ArgList[0]); err != nil {
		return Value{}, err
	}
	t := e.h.TypeOf(c.ArgList[0])
	if t == nil {
		return Value{}, e.fail(pos, "unsupported make type in const evaluation")
	}
	if t.Impure != "" {
		return Value{}, e.impurity(fr, pos, "%s", t.Impure)
	}
	intArg := func(i int) (int64, error) {
		v, err := e.evalExpr(fr, c.ArgList[i])
		if err != nil {
			return 0, err
		}
		if !isInteger(kindOf(v)) {
			return 0, e.fail(pos, "non-integer make argument in const evaluation")
		}
		return v.ival.Int64(), nil
	}
	switch t.Kind {
	case KSlice:
		var ln, cp int64
		var err error
		if len(c.ArgList) >= 2 {
			if ln, err = intArg(1); err != nil {
				return Value{}, err
			}
		}
		cp = ln
		if len(c.ArgList) >= 3 {
			if cp, err = intArg(2); err != nil {
				return Value{}, err
			}
		}
		if len(c.ArgList) > 3 || c.HasDots {
			return Value{}, e.fail(pos, "bad make call in const evaluation")
		}
		if ln < 0 || cp < ln {
			return Value{}, e.fail(pos, "negative make length in const evaluation")
		}
		if ln > 0 && cp > e.cfg.MaxValueBytes {
			return Value{}, &Error{Code: CodeTooLarge, Detail: fmt.Sprintf("make of %d elements", cp), Pos: pos}
		}
		elems := make([]Value, ln)
		for i := range elems {
			elems[i] = zeroValue(t.Elem)
		}
		backing := append([]Value(nil), elems...)
		for len(backing) < int(cp) {
			backing = append(backing, zeroValue(t.Elem))
		}
		return Value{typ: t, slice: &sliceVal{backing: &backing, length: int(ln), cap_: int(cp)}}, nil
	case KMap:
		if len(c.ArgList) > 2 || c.HasDots {
			return Value{}, e.fail(pos, "bad make call in const evaluation")
		}
		return Value{typ: t, mval: &mapVal{}}, nil
	}
	return Value{}, e.fail(pos, "bad make call in const evaluation")
}

// evalExtremum folds min (op Lss) or max (op Gtr) over ordered operands.
func (e *evaluator) evalExtremum(op syntax.Operator, fr *frame, c *syntax.CallExpr) (Value, error) {
	pos := posOf(c)
	if len(c.ArgList) < 1 {
		return Value{}, e.fail(pos, "bad min/max call in const evaluation")
	}
	best, err := e.evalExpr(fr, c.ArgList[0])
	if err != nil {
		return Value{}, err
	}
	for _, a := range c.ArgList[1:] {
		v, err := e.evalExpr(fr, a)
		if err != nil {
			return Value{}, err
		}
		win, err := e.compare(op, v, best, pos)
		if err != nil {
			return Value{}, err
		}
		if win.bval {
			best = v
		}
	}
	return best, nil
}

// checkSize enforces the value cap over live frames plus extras.
func (e *evaluator) checkSize(extras ...Value) error {
	seen := map[any]bool{}
	var total int64
	add := func(v Value) error {
		total += e.valueSize(v, seen)
		if total > e.cfg.MaxValueBytes {
			return &Error{Code: CodeTooLarge, Detail: fmt.Sprintf("%d bytes over %d byte limit", total, e.cfg.MaxValueBytes)}
		}
		return nil
	}
	for _, fr := range e.frames {
		for _, sc := range fr.scopes {
			for _, v := range sc {
				if err := add(v); err != nil {
					return err
				}
			}
		}
	}
	for _, v := range extras {
		if err := add(v); err != nil {
			return err
		}
	}
	return nil
}

// valueSize counts payload bytes with sharing deduplicated.
func (e *evaluator) valueSize(v Value, seen map[any]bool) int64 {
	switch {
	case v.ival != nil:
		if v.typ != nil && isInteger(v.typ.Kind) && !isUntyped(v.typ.Kind) {
			if w := intWidth(v.typ.Kind, e.cfg.WordSize) / 8; w > 0 {
				return w
			}
		}
		return 8
	case v.fval != nil:
		return 16
	case v.typ != nil && (v.typ.Kind == KFloat32 || v.typ.Kind == KFloat64):
		return 8
	case v.sval != "" || (v.typ != nil && v.typ.Kind == KString):
		return int64(len(v.sval))
	case v.slice != nil:
		if v.isNil || v.slice.backing == nil {
			return 0
		}
		var n int64
		if !seen[v.slice.backing] {
			seen[v.slice.backing] = true
			for _, el := range sliceElems(v) {
				n += e.valueSize(el, seen)
			}
		}
		return n
	case v.array != nil:
		var n int64
		for _, el := range v.array {
			n += e.valueSize(el, seen)
		}
		return n
	case v.strct != nil:
		var n int64
		for _, f := range v.strct {
			n += e.valueSize(f, seen)
		}
		return n
	case v.mval != nil:
		if v.isNil {
			return 0
		}
		if seen[v.mval] {
			return 0
		}
		seen[v.mval] = true
		var n int64
		for i, k := range v.mval.keys {
			n += e.valueSize(k, seen) + e.valueSize(v.mval.vals[i], seen)
		}
		return n
	}
	return 1
}
