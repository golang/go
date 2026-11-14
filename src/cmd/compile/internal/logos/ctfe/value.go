// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ctfe

import (
	"cmd/compile/internal/syntax"
	"fmt"
	"math"
	"math/big"
	"slices"
	"strconv"
	"unicode/utf8"
)

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

func BoolValue(b bool, t *Type) Value { return Value{typ: t, constant: true, bval: b} }

func StringValue(s string, t *Type) Value { return Value{typ: t, constant: true, sval: s} }

func IntValue(n *big.Int, t *Type) Value { return Value{typ: t, constant: true, ival: n} }

func FloatValue(f float64, t *Type) Value { return Value{typ: t, constant: true, f64: f} }

func ComplexValue(c complex128, t *Type) Value { return Value{typ: t, constant: true, cval: c} }

func ExactFloatValue(r *big.Rat, t *Type) Value {
	var copyR *big.Rat
	if r != nil {
		copyR = new(big.Rat).Set(r)
	}
	return Value{typ: t, constant: true, fval: copyR}
}

func defaulted(v Value) Value {
	if v.typ == nil || !isUntyped(v.typ.Kind) {
		return v
	}
	if v.typ.Kind == KUntypedComplex {
		v.cval = v.Complex()
		v.cparts = nil
	}
	v.typ = defaultType(v.typ.Kind)
	return v
}

func zeroValue(t *Type) Value {
	switch t.Kind {
	case KBool, KUntypedBool:
		return Value{typ: &Type{Kind: KBool}, bval: false}
	case KFloat32, KFloat64, KUntypedFloat:
		k := t.Kind
		if k == KUntypedFloat {
			k = KFloat64
		}
		return Value{typ: &Type{Kind: k}, f64: 0}
	case KComplex64, KComplex128, KUntypedComplex:
		k := t.Kind
		if k == KUntypedComplex {
			k = KComplex128
		}
		return Value{typ: &Type{Kind: k}, cval: 0}
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
		return Value{typ: &Type{Kind: k}, ival: new(big.Int)}
	}
	if t.Kind == KInvalid && t.Impure != "" {
		return Value{typ: t, isNil: true}
	}
	return Value{typ: t}
}

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

type storeKey struct {
	first *Value
	n     int
}

type hostCopy struct {
	e        *evaluator
	pos      syntax.Pos
	slots    int64
	bytes    int64
	stores   map[storeKey][]Value
	backings map[*[]Value]*[]Value
	maps     map[*mapVal]*mapVal
}

func (c *hostCopy) charge(slots int, payload int64) error {
	c.slots += int64(slots)
	c.bytes += payload
	limit := c.e.cfg.MaxValueBytes
	if c.slots > limit {
		return &Error{Code: CodeTooLarge, Detail: fmt.Sprintf("copy of %d elements over %d byte limit", c.slots, limit), Pos: c.pos}
	}
	if c.bytes > limit {
		return &Error{Code: CodeTooLarge, Detail: fmt.Sprintf("%d bytes over %d byte limit", c.bytes, limit), Pos: c.pos}
	}
	return nil
}

func (c *hostCopy) value(v Value) (Value, error) {
	switch {
	case v.array != nil:
		s, err := c.store(v.array)
		if err != nil {
			return Value{}, err
		}
		v.array = s
	case v.strct != nil:
		s, err := c.store(v.strct)
		if err != nil {
			return Value{}, err
		}
		v.strct = s
	case v.slice != nil:
		h := *v.slice
		if h.backing != nil {
			b, err := c.backing(h.backing)
			if err != nil {
				return Value{}, err
			}
			h.backing = b
		}
		v.slice = &h
	case v.mval != nil:
		m, err := c.mapOf(v.mval)
		if err != nil {
			return Value{}, err
		}
		v.mval = m
	default:
		if err := c.charge(0, c.e.valueSize(v, nil)); err != nil {
			return Value{}, err
		}
	}
	return v, nil
}

func (c *hostCopy) store(src []Value) ([]Value, error) {
	if len(src) == 0 {
		return src, nil
	}
	key := storeKey{&src[0], len(src)}
	if dst, ok := c.stores[key]; ok {
		return dst, nil
	}
	if err := c.e.chargeSteps(1, c.pos); err != nil {
		return nil, err
	}
	if err := c.charge(len(src), 0); err != nil {
		return nil, err
	}
	dst := make([]Value, len(src))
	c.stores[key] = dst
	for i := range src {
		v, err := c.value(src[i])
		if err != nil {
			return nil, err
		}
		dst[i] = v
	}
	return dst, nil
}

func (c *hostCopy) backing(src *[]Value) (*[]Value, error) {
	if dst, ok := c.backings[src]; ok {
		return dst, nil
	}
	dst := new([]Value)
	c.backings[src] = dst
	s, err := c.store(*src)
	if err != nil {
		return nil, err
	}
	*dst = s
	return dst, nil
}

func (c *hostCopy) mapOf(src *mapVal) (*mapVal, error) {
	if dst, ok := c.maps[src]; ok {
		return dst, nil
	}
	if err := c.e.chargeSteps(1, c.pos); err != nil {
		return nil, err
	}
	if err := c.charge(len(src.keys)+len(src.vals), 0); err != nil {
		return nil, err
	}
	dst := &mapVal{keys: make([]Value, len(src.keys)), vals: make([]Value, len(src.vals))}
	c.maps[src] = dst
	for i := range src.keys {
		k, err := c.value(src.keys[i])
		if err != nil {
			return nil, err
		}
		dst.keys[i] = k
	}
	for i := range src.vals {
		v, err := c.value(src.vals[i])
		if err != nil {
			return nil, err
		}
		dst.vals[i] = v
	}
	return dst, nil
}

func (e *evaluator) ingest(v Value, pos syntax.Pos) (Value, error) {
	if v.array == nil && v.strct == nil && v.slice == nil && v.mval == nil {
		return v, nil
	}
	c := &hostCopy{
		e:        e,
		pos:      pos,
		stores:   map[storeKey][]Value{},
		backings: map[*[]Value]*[]Value{},
		maps:     map[*mapVal]*mapVal{},
	}
	out, err := c.value(v)
	if err != nil {
		return Value{}, err
	}
	if _, serr := e.checkSize(out); serr != nil {
		serr.Pos = pos
		return Value{}, serr
	}
	return out, nil
}

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

func (e *evaluator) evalConversion(fr *frame, c *syntax.CallExpr) (Value, error) {
	if err := e.precheckType(fr, c.Fun); err != nil {
		return Value{}, err
	}
	t := e.h.TypeOf(c)
	if t == nil {
		t = e.h.TypeOf(c.Fun)
	}
	if t == nil {
		t = e.h.TypeOf(syntax.Unparen(c.Fun))
	}
	if t == nil {
		var err error
		t, err = e.resolveType(fr, c.Fun)
		if err != nil {
			return Value{}, err
		}
	}
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

func (e *evaluator) convert(v Value, t *Type, pos syntax.Pos) (Value, error) {
	if t.Kind == KUintptr {
		return Value{}, e.impurity(nil, pos, "conversion to uintptr in const func")
	}
	if v.isNil {
		return e.coerceNil(v, t, pos)
	}
	vk := kindOf(v)
	if v.typ != nil && v.typ.Kind == t.Kind {
		v.typ = t
		return v, nil
	}
	switch t.Kind {
	case KString:
		return e.convertToString(v, t, pos)
	case KSlice:
		if isStringLike(vk) && t.Elem != nil && (t.Elem.Kind == KUint8 || t.Elem.Kind == KInt32) {
			return e.stringToSlice(v, t, pos)
		}
	}
	if !isUntyped(vk) && isStringLike(vk) && t.Kind == KSlice {
		return Value{}, e.fail(pos, "cannot convert string in const evaluation")
	}
	if isInteger(t.Kind) || isFloat(t.Kind) {
		return e.convertNumber(v, t, pos)
	}
	if isComplex(t.Kind) {
		return e.convertComplex(v, t, pos)
	}
	if t.Kind == KBool && (vk == KBool || vk == KUntypedBool) {
		v.typ = t
		return v, nil
	}
	return Value{}, e.fail(pos, "cannot convert %s in const evaluation", typeName(v.typ))
}

func (e *evaluator) coerceNil(v Value, t *Type, pos syntax.Pos) (Value, error) {
	switch t.Kind {
	case KSlice, KMap:
		return Value{typ: t, isNil: true}, nil
	}
	return Value{}, e.fail(pos, "cannot use nil as %s in const evaluation", typeName(t))
}

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
		if t.Kind == KFloat32 && math.IsInf(float64(float32(f)), 0) {
			return Value{}, e.fail(pos, "constant overflows float32 in const evaluation")
		}
		if t.Kind == KFloat64 && math.IsInf(f, 0) {
			return Value{}, e.fail(pos, "constant overflows float64 in const evaluation")
		}
		if isUntyped(t.Kind) {
			if xf := xFloat(v); xf != nil {
				return Value{typ: t, constant: true, fval: new(big.Rat).Set(xf)}, nil
			}
			return Value{}, e.fail(pos, "constant overflows %s in const evaluation", typeName(t))
		}
		return floatResult(t, f, true, pos)
	}
	return floatResult(t, f, false, pos)
}

func (e *evaluator) convertComplex(v Value, t *Type, pos syntax.Pos) (Value, error) {
	vk := kindOf(v)
	if !isNumeric(vk) {
		return Value{}, e.fail(pos, "cannot convert %s to %s in const evaluation", typeName(v.typ), typeName(t))
	}
	if v.constant && t.Kind == KUntypedComplex {
		re, im := complexParts(v)
		parts := new([2]big.Rat)
		parts[0].Set(re)
		parts[1].Set(im)
		return Value{typ: t, constant: true, cparts: parts}, nil
	}
	var c complex128
	switch {
	case v.constant && isUntyped(vk):
		re, im := complexParts(v)
		if t.Kind == KComplex64 {
			rf, _ := re.Float32()
			imf, _ := im.Float32()
			c = complex(float64(rf), float64(imf))
		} else {
			rf, _ := re.Float64()
			imf, _ := im.Float64()
			c = complex(rf, imf)
		}
	case isComplex(vk):
		c = v.Complex()
	case isFloat(vk):
		c = complex(v.Float64(), 0)
	case isInteger(vk):
		f, _ := new(big.Rat).SetInt(v.ival).Float64()
		c = complex(f, 0)
	}
	if v.constant && (math.IsInf(real(c), 0) || math.IsInf(imag(c), 0)) {
		if t.Kind == KComplex64 {
			return Value{}, e.fail(pos, "constant overflows complex64 in const evaluation")
		}
		return Value{}, e.fail(pos, "constant overflows %s in const evaluation", typeName(t))
	}
	if t.Kind == KComplex64 {
		if v.constant && (math.IsInf(float64(float32(real(c))), 0) || math.IsInf(float64(float32(imag(c))), 0)) {
			return Value{}, e.fail(pos, "constant overflows complex64 in const evaluation")
		}
		c = complex(float64(float32(real(c))), float64(float32(imag(c))))
	}
	return Value{typ: t, constant: v.constant, cval: c}, nil
}

func toFloatParts(v Value, t *Type) float64 {
	if t.Kind == KComplex64 && v.constant && (v.fval != nil || v.ival != nil) {
		f, _ := xFloat(v).Float32()
		return float64(f)
	}
	if isFloat(kindOf(v)) {
		return v.Float64()
	}
	f, _ := new(big.Rat).SetInt(v.ival).Float64()
	return f
}

func (e *evaluator) convertToString(v Value, t *Type, pos syntax.Pos) (Value, error) {
	vk := kindOf(v)
	if vk == KString || vk == KUntypedString {
		return Value{typ: t, constant: v.constant, sval: v.sval}, nil
	}
	if vk == KSlice {
		return e.sliceToString(v, t, pos)
	}
	if !isInteger(vk) {
		return Value{}, e.fail(pos, "cannot convert %s to string in const evaluation", typeName(v.typ))
	}
	if v.ival == nil {
		return Value{}, e.fail(pos, "cannot convert %s to string in const evaluation", typeName(v.typ))
	}
	if v.ival.Sign() < 0 || !v.ival.IsInt64() {
		return Value{typ: t, constant: v.constant, sval: "\uFFFD"}, nil
	}
	n := v.ival.Int64()
	s := string(rune(n))
	if n > utf8.MaxRune || (n >= 0xD800 && n <= 0xDFFF) {
		s = "\uFFFD"
	}
	return Value{typ: t, constant: v.constant, sval: s}, nil
}

func (e *evaluator) sliceToString(v Value, t *Type, pos syntax.Pos) (Value, error) {
	if v.typ == nil || v.typ.Elem == nil || v.isNil || v.slice == nil || v.slice.backing == nil {
		return Value{typ: t, constant: v.constant, sval: ""}, nil
	}
	switch v.typ.Elem.Kind {
	case KUint8:
		buf := make([]byte, 0, v.slice.length)
		for _, el := range sliceElems(v) {
			if el.ival == nil {
				return Value{}, e.fail(pos, "cannot convert %s to string in const evaluation", typeName(v.typ))
			}
			buf = append(buf, byte(el.ival.Int64()))
		}
		return Value{typ: t, constant: v.constant, sval: string(buf)}, nil
	case KInt32:
		runes := make([]rune, 0, v.slice.length)
		for _, el := range sliceElems(v) {
			if el.ival == nil {
				return Value{}, e.fail(pos, "cannot convert %s to string in const evaluation", typeName(v.typ))
			}
			runes = append(runes, rune(el.ival.Int64()))
		}
		return Value{typ: t, constant: v.constant, sval: string(runes)}, nil
	}
	return Value{}, e.fail(pos, "cannot convert %s to string in const evaluation", typeName(v.typ))
}

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
		for len(elems) <= next {
			elems = append(elems, zeroValue(elemT))
		}
		v, err := e.evalElem(fr, el, elemT)
		if err != nil {
			return nil, err
		}
		elems[next] = v
		next++
	}
	for i := range elems {
		elems[i] = copyValue(withVar(elems[i]))
	}
	return elems, nil
}

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
	return e.convertElem(v, elemT, posOf(el))
}

func (e *evaluator) convertElem(v Value, t *Type, pos syntax.Pos) (Value, error) {
	if t == nil {
		return v, nil
	}
	if v.isNil {
		return e.coerceNil(v, t, pos)
	}
	return e.convert(v, t, pos)
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
		k, err := e.evalElem(fr, kv.Key, t.Key)
		if err != nil {
			return Value{}, err
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

func (e *evaluator) mapInsert(m, k, v Value, pos syntax.Pos) error {
	if m.isNil {
		return e.fail(pos, "assignment to nil map in const evaluation")
	}
	if isNaNKey(k) {
		return &Error{Code: CodeEval, Detail: "NaN is not a valid const map key", Pos: pos}
	}
	k = e.canonKey(k)
	for _, ek := range m.mval.keys {
		eq, err := e.equal(ek, k, pos)
		if err != nil {
			return err
		}
		if eq {
			return &Error{Code: CodeEval, Detail: "duplicate key in const map", Pos: pos}
		}
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
	if isComplex(k.typ.Kind) {
		c := k.Complex()
		return math.IsNaN(real(c)) || math.IsNaN(imag(c))
	}
	switch k.typ.Kind {
	case KArray:
		for _, elem := range k.array {
			if isNaNKey(elem) {
				return true
			}
		}
	case KStruct:
		for _, field := range k.strct {
			if isNaNKey(field) {
				return true
			}
		}
	}
	return false
}

func (e *evaluator) canonKey(v Value) Value {
	if v.typ != nil && (v.typ.Kind == KFloat32 || v.typ.Kind == KFloat64) && !v.constant {
		if v.f64 == 0 {
			v.f64 = 0
		}
	}
	if v.typ != nil && isComplex(v.typ.Kind) && v.cparts == nil {
		if real(v.cval) == 0 {
			v.cval = complex(0, imag(v.cval))
		}
		if imag(v.cval) == 0 {
			v.cval = complex(real(v.cval), 0)
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

func (e *evaluator) evalSelector(fr *frame, sel *syntax.SelectorExpr) (Value, error) {
	if name, ok := sel.X.(*syntax.Name); ok {
		if ent := e.h.Resolve(name); ent.Kind == EntPackage {
			rent := e.h.ResolveSel(sel)
			switch rent.Kind {
			case EntConst:
				return e.ingest(rent.Const, posOf(sel))
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
	case x.typ != nil && isStringLike(x.typ.Kind):
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

func (e *evaluator) indexPlace(fr *frame, ix *syntax.IndexExpr) (place, error) {
	if _, ok := ix.Index.(*syntax.ListExpr); ok {
		return place{}, e.fail(posOf(ix), "generic instantiation in const func not supported")
	}
	k, err := e.evalExpr(fr, ix.Index)
	if err != nil {
		return place{}, err
	}
	if base, ok := syntax.Unparen(ix.X).(*syntax.Name); ok {
		if bv, found := e.lookup(fr, base.Value); found && bv.mval != nil {
			var keyT *Type
			if bv.typ != nil {
				keyT = bv.typ.Key
			}
			if k, err = e.convertElem(k, keyT, posOf(ix.Index)); err != nil {
				return place{}, err
			}
			if isNaNKey(k) {
				return place{}, &Error{Code: CodeEval, Detail: "NaN is not a valid const map key", Pos: posOf(ix)}
			}
			for i := len(fr.scopes) - 1; i >= 0; i-- {
				if _, ok := fr.scopes[i][base.Value]; ok {
					idx := i
					name := base.Value
					return place{
						get: func() Value {
							curMap := fr.scopes[idx][name]
							v, err := e.indexValue(curMap, k, posOf(ix))
							if err != nil {
								if curMap.typ != nil && curMap.typ.Elem != nil {
									return zeroValue(curMap.typ.Elem)
								}
								return zeroValue(&Type{Kind: KInt})
							}
							return v
						},
						set: func(v Value) {
							cur := fr.scopes[idx][name]
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
		if lo < 0 || hi < lo || hi > c || max < hi || max > c {
			return Value{}, e.fail(posOf(sx), "slice bounds out of range in const evaluation")
		}
		nv := x
		nh := *x.slice
		nh.lo += int(lo)
		nh.length = int(hi - lo)
		nh.cap_ = x.slice.lo + int(max)
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
		backing := x.array
		return Value{
			typ:      &Type{Kind: KSlice, Elem: x.typ.Elem},
			slice:    &sliceVal{backing: &backing, lo: int(lo), length: int(hi - lo), cap_: int(n)},
			constant: false,
		}, nil
	case x.typ != nil && isStringLike(x.typ.Kind):
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
		if v.isNil && v.typ != nil && (v.typ.Kind == KSlice || v.typ.Kind == KMap) {
			if v.typ.Kind == KSlice {
				return intValue(big.NewInt(0), UntypedInt(), true), nil
			}
			if name == "len" {
				return intValue(big.NewInt(0), UntypedInt(), true), nil
			}
		}
		switch {
		case v.typ != nil && isStringLike(v.typ.Kind):
			if name == "len" {
				return intValue(big.NewInt(int64(len(v.sval))), UntypedInt(), true), nil
			}
			return Value{}, e.fail(pos, "bad cap call on string in const evaluation")
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
			for i := range v.slice.length {
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
	case "complex":
		if len(c.ArgList) != 2 || c.HasDots {
			return Value{}, e.fail(pos, "bad complex call in const evaluation")
		}
		re, err := e.evalExpr(fr, c.ArgList[0])
		if err != nil {
			return Value{}, err
		}
		im, err := e.evalExpr(fr, c.ArgList[1])
		if err != nil {
			return Value{}, err
		}
		if !isFloat(kindOf(re)) && !(isInteger(kindOf(re)) && isUntyped(kindOf(re))) {
			return Value{}, e.fail(pos, "bad complex call in const evaluation")
		}
		if !isFloat(kindOf(im)) && !(isInteger(kindOf(im)) && isUntyped(kindOf(im))) {
			return Value{}, e.fail(pos, "bad complex call in const evaluation")
		}
		t := UntypedComplex()
		if !isUntyped(kindOf(re)) || !isUntyped(kindOf(im)) {
			t = &Type{Kind: KComplex128}
			if kindOf(re) == KFloat32 || kindOf(im) == KFloat32 {
				t = &Type{Kind: KComplex64}
			}
		}
		if re.constant && im.constant {
			if t.Kind == KUntypedComplex {
				parts := new([2]big.Rat)
				parts[0].Set(xFloat(re))
				parts[1].Set(xFloat(im))
				return Value{typ: t, constant: true, cparts: parts}, nil
			}
			rf, imf := toFloatParts(re, t), toFloatParts(im, t)
			return complexConst(t, complex(rf, imf)), nil
		}
		rf, imf := toFloatParts(re, t), toFloatParts(im, t)
		return complexDyn(t, complex(rf, imf)), nil
	case "real", "imag":
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

func (e *evaluator) evalAppend(fr *frame, c *syntax.CallExpr) (Value, error) {
	pos := posOf(c)
	if len(c.ArgList) < 1 {
		return Value{}, e.fail(pos, "bad append call in const evaluation")
	}
	s, err := e.evalExpr(fr, c.ArgList[0])
	if err != nil {
		return Value{}, err
	}
	if s.isNil && s.typ != nil && s.typ.Kind == KSlice {
		if len(c.ArgList) == 1 {
			return s, nil
		}
		s.isNil = false
		s.slice = &sliceVal{backing: new([]Value), length: 0, cap_: 0}
	} else if s.slice == nil {
		return Value{}, e.fail(pos, "append to non-slice in const evaluation")
	}
	elems, err := e.evalAppendArgs(fr, &syntax.CallExpr{ArgList: c.ArgList[1:], HasDots: c.HasDots}, s)
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
		if int64(length+len(elems)) > e.cfg.MaxValueBytes {
			return Value{}, &Error{Code: CodeTooLarge, Detail: fmt.Sprintf("append of %d elements over %d byte limit", length+len(elems), e.cfg.MaxValueBytes), Pos: pos}
		}
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
	if int64(need) > e.cfg.MaxValueBytes || int64(newcap) > e.cfg.MaxValueBytes {
		return Value{}, &Error{Code: CodeTooLarge, Detail: fmt.Sprintf("append of %d elements over %d byte limit", need, e.cfg.MaxValueBytes), Pos: pos}
	}
	if _, err := e.checkSize(); err != nil {
		err.Pos = pos
		return Value{}, err
	}
	backing := make([]Value, newcap)
	copy(backing, sliceElems(s))
	copy(backing[length:], elems)
	s.slice = &sliceVal{backing: &backing, length: need, cap_: newcap}
	return s, nil
}

func (e *evaluator) evalAppendArgs(fr *frame, c *syntax.CallExpr, dst Value) ([]Value, error) {
	var out []Value
	for i, a := range c.ArgList {
		v, err := e.evalExpr(fr, a)
		if err != nil {
			return nil, err
		}
		if c.HasDots && i == len(c.ArgList)-1 {
			if v.typ != nil && isStringLike(v.typ.Kind) {
				if dst.typ == nil || dst.typ.Kind != KSlice || dst.typ.Elem == nil || dst.typ.Elem.Kind != KUint8 {
					return nil, e.fail(posOf(a), "cannot spread string in const evaluation")
				}
				for _, b := range []byte(v.sval) {
					out = append(out, byteValue(b))
				}
				continue
			}
			if v.slice == nil && !(v.isNil && v.typ != nil && v.typ.Kind == KSlice) {
				return nil, e.fail(posOf(a), "cannot spread non-slice in const evaluation")
			}
			if v.slice != nil {
				out = append(out, sliceElems(v)...)
			}
			continue
		}
		out = append(out, v)
	}
	return out, nil
}

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
	snap := slices.Clone(selems)
	delems := sliceElems(dst)
	n := min(len(delems), len(snap))
	for i := range n {
		v := copyValue(snap[i])
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
		if cp > e.cfg.MaxValueBytes {
			return Value{}, &Error{Code: CodeTooLarge, Detail: fmt.Sprintf("make of %d elements", cp), Pos: pos}
		}
		backing := make([]Value, cp)
		for i := range backing {
			backing[i] = zeroValue(t.Elem)
		}
		return Value{typ: t, slice: &sliceVal{backing: &backing, length: int(ln), cap_: int(cp)}}, nil
	case KMap:
		if len(c.ArgList) > 2 || c.HasDots {
			return Value{}, e.fail(pos, "bad make call in const evaluation")
		}
		if len(c.ArgList) == 2 {
			if _, err := intArg(1); err != nil {
				return Value{}, err
			}
		}
		return Value{typ: t, mval: &mapVal{}}, nil
	}
	return Value{}, e.fail(pos, "bad make call in const evaluation")
}

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

func (e *evaluator) checkSize(extras ...Value) (int64, *Error) {
	seen := map[any]bool{}
	var total int64
	add := func(v Value) *Error {
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
					return total, err
				}
			}
		}
	}
	for _, v := range extras {
		if err := add(v); err != nil {
			return total, err
		}
	}
	return total, nil
}

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
	case v.typ != nil && (v.typ.Kind == KComplex64 || v.typ.Kind == KComplex128 || v.typ.Kind == KUntypedComplex):
		return 16
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
