// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ctfe

import (
	"cmd/compile/internal/syntax"
	"fmt"
	"math"
	"math/big"
	"strconv"
	"strings"
)

func (v Value) Expr(pos syntax.Pos) (syntax.Expr, error) {
	r := &renderer{pos: pos}
	x, err := r.expr(v)
	if err != nil {
		return nil, err
	}
	syntax.Inspect(x, func(n syntax.Node) bool {
		if n != nil {
			n.SetPos(pos)
		}
		return true
	})
	return x, nil
}

type renderer struct {
	pos syntax.Pos
}

func (r *renderer) expr(v Value) (syntax.Expr, error) {
	if v.isNil {
		if v.typ == nil || v.typ.Kind == KUntypedNil {
			return syntax.NewName(r.pos, "nil"), nil
		}
		t, err := r.typeExpr(v.typ)
		if err != nil {
			return nil, err
		}
		return &syntax.CallExpr{Fun: t, ArgList: []syntax.Expr{syntax.NewName(r.pos, "nil")}}, nil
	}
	switch {
	case v.ival != nil:
		return r.intLit(v)
	case v.fval != nil || (v.typ != nil && isFloat(v.typ.Kind)):
		return r.floatLit(v)
	case v.typ != nil && isComplex(v.typ.Kind):
		return r.complexLit(v)
	case v.typ != nil && (v.typ.Kind == KString || v.typ.Kind == KUntypedString):
		return &syntax.BasicLit{Value: strconv.Quote(v.sval), Kind: syntax.StringLit}, nil
	case v.typ != nil && (v.typ.Kind == KBool || v.typ.Kind == KUntypedBool):
		name := "false"
		if v.bval {
			name = "true"
		}
		return syntax.NewName(r.pos, name), nil
	case v.slice != nil:
		return r.sliceLit(v)
	case v.array != nil:
		return r.arrayLit(v)
	case v.strct != nil:
		return r.structLit(v)
	case v.mval != nil:
		return r.mapLit(v)
	}
	return nil, fmt.Errorf("cannot spell value in const evaluation")
}

func (r *renderer) intLit(v Value) (syntax.Expr, error) {
	s := v.ival.String()
	lit := &syntax.BasicLit{Value: s, Kind: syntax.IntLit}
	if v.ival.Sign() < 0 {
		mag := &syntax.BasicLit{Value: s[1:], Kind: syntax.IntLit}
		return &syntax.Operation{Op: syntax.Sub, X: mag}, nil
	}
	return lit, nil
}

func (r *renderer) floatLit(v Value) (syntax.Expr, error) {
	var (
		s   string
		neg bool
	)
	if v.fval != nil {
		f, _ := v.fval.Float64()
		if v.typ != nil && v.typ.Kind == KFloat32 {
			f = float64(float32(f))
		}
		s, neg = r.floatText(f)
	} else {
		s, neg = r.floatText(v.f64)
	}
	lit := &syntax.BasicLit{Value: s, Kind: syntax.FloatLit}
	if neg {
		return &syntax.Operation{Op: syntax.Sub, X: lit}, nil
	}
	return lit, nil
}

func (r *renderer) complexLit(v Value) (syntax.Expr, error) {
	if v.constant && kindOf(v) == KUntypedComplex {
		re, im := complexParts(v)
		if re == nil || im == nil {
			return nil, fmt.Errorf("cannot spell non-finite complex constant in const evaluation")
		}
		return &syntax.CallExpr{
			Fun:     syntax.NewName(r.pos, "complex"),
			ArgList: []syntax.Expr{r.ratLit(re), r.ratLit(im)},
		}, nil
	}
	c := v.Complex()
	re, im := real(c), imag(c)
	if v.typ != nil && v.typ.Kind == KComplex64 {
		re = float64(float32(re))
		im = float64(float32(im))
	}
	rs, rneg := r.floatText(re)
	is, ineg := r.floatText(im)
	var rexp, iexp syntax.Expr = &syntax.BasicLit{Value: rs, Kind: syntax.FloatLit}, &syntax.BasicLit{Value: is, Kind: syntax.FloatLit}
	if rneg {
		rexp = &syntax.Operation{Op: syntax.Sub, X: rexp}
	}
	if ineg {
		iexp = &syntax.Operation{Op: syntax.Sub, X: iexp}
	}
	return &syntax.CallExpr{Fun: syntax.NewName(r.pos, "complex"), ArgList: []syntax.Expr{rexp, iexp}}, nil
}

func (r *renderer) ratLit(v *big.Rat) syntax.Expr {
	s := v.Num().String()
	var x syntax.Expr = &syntax.BasicLit{Value: strings.TrimPrefix(s, "-") + ".0", Kind: syntax.FloatLit}
	if v.Sign() < 0 {
		x = &syntax.Operation{Op: syntax.Sub, X: x}
	}
	if !v.IsInt() {
		x = &syntax.Operation{Op: syntax.Div, X: x, Y: &syntax.BasicLit{Value: v.Denom().String() + ".0", Kind: syntax.FloatLit}}
	}
	return x
}

func (r *renderer) floatText(f float64) (string, bool) {
	if math.Signbit(f) {
		if f == 0 {
			return "0.0", true
		}
		s := strconv.FormatFloat(-f, 'g', -1, 64)
		if !strings.ContainsAny(s, ".eE") {
			s += ".0"
		}
		return s, true
	}
	s := strconv.FormatFloat(f, 'g', -1, 64)
	if !strings.ContainsAny(s, ".eE") {
		s += ".0"
	}
	return s, false
}

func (r *renderer) sliceLit(v Value) (syntax.Expr, error) {
	t, err := r.typeExpr(v.typ)
	if err != nil {
		return nil, err
	}
	if v.SliceNeedsBacking() {
		return r.backedSliceLit(v, t)
	}
	elems := make([]syntax.Expr, 0, len(sliceElems(v)))
	for _, el := range sliceElems(v) {
		x, err := r.expr(el)
		if err != nil {
			return nil, err
		}
		elems = append(elems, x)
	}
	return &syntax.CompositeLit{Type: t, ElemList: elems, Rbrace: r.pos}, nil
}

func (r *renderer) backedSliceLit(v Value, t syntax.Expr) (syntax.Expr, error) {
	backing, lo, length, cap_, ok := v.SliceHeader()
	if !ok || lo < 0 || length < 0 || lo > cap_ || lo+length > cap_ || cap_ > len(backing) {
		return nil, fmt.Errorf("cannot spell slice window in const evaluation")
	}
	elems := make([]syntax.Expr, 0, cap_-lo)
	for _, el := range backing[lo:cap_] {
		x, err := r.expr(el)
		if err != nil {
			return nil, err
		}
		elems = append(elems, x)
	}
	backingLit := &syntax.CompositeLit{Type: t, ElemList: elems, Rbrace: r.pos}
	return &syntax.SliceExpr{
		X:     backingLit,
		Index: [3]syntax.Expr{r.sliceBound(0), r.sliceBound(length), r.sliceBound(cap_ - lo)},
		Full:  true,
	}, nil
}

func (r *renderer) sliceBound(n int) syntax.Expr {
	return &syntax.BasicLit{Value: strconv.Itoa(n), Kind: syntax.IntLit}
}

func (r *renderer) arrayLit(v Value) (syntax.Expr, error) {
	t, err := r.typeExpr(v.typ)
	if err != nil {
		return nil, err
	}
	if at, ok := t.(*syntax.ArrayType); ok {
		at.Len = &syntax.BasicLit{Value: strconv.Itoa(len(v.array)), Kind: syntax.IntLit}
	}
	elems := make([]syntax.Expr, 0, len(v.array))
	for _, el := range v.array {
		x, err := r.expr(el)
		if err != nil {
			return nil, err
		}
		elems = append(elems, x)
	}
	return &syntax.CompositeLit{Type: t, ElemList: elems, Rbrace: r.pos}, nil
}

func (r *renderer) structLit(v Value) (syntax.Expr, error) {
	t, err := r.typeExpr(v.typ)
	if err != nil {
		return nil, err
	}
	elems := make([]syntax.Expr, 0, len(v.strct))
	for i, f := range v.typ.Fields {
		x, err := r.expr(v.strct[i])
		if err != nil {
			return nil, err
		}
		elems = append(elems, &syntax.KeyValueExpr{Key: syntax.NewName(r.pos, f.Name), Value: x})
	}
	return &syntax.CompositeLit{Type: t, ElemList: elems, NKeys: len(elems), Rbrace: r.pos}, nil
}

func (r *renderer) mapLit(v Value) (syntax.Expr, error) {
	t, err := r.typeExpr(v.typ)
	if err != nil {
		return nil, err
	}
	elems := make([]syntax.Expr, 0, len(v.mval.keys))
	for i, k := range v.mval.keys {
		kx, err := r.expr(k)
		if err != nil {
			return nil, err
		}
		vx, err := r.expr(v.mval.vals[i])
		if err != nil {
			return nil, err
		}
		elems = append(elems, &syntax.KeyValueExpr{Key: kx, Value: vx})
	}
	return &syntax.CompositeLit{Type: t, ElemList: elems, NKeys: len(elems), Rbrace: r.pos}, nil
}

func (r *renderer) typeExpr(t *Type) (syntax.Expr, error) {
	if t == nil {
		return nil, fmt.Errorf("missing type in const evaluation")
	}
	if t.Opaque {
		return nil, fmt.Errorf("cannot spell type in const evaluation")
	}
	if t.Named != "" {
		return syntax.NewName(r.pos, t.Named), nil
	}
	switch t.Kind {
	case KBool:
		return syntax.NewName(r.pos, "bool"), nil
	case KInt:
		return syntax.NewName(r.pos, "int"), nil
	case KInt8:
		return syntax.NewName(r.pos, "int8"), nil
	case KInt16:
		return syntax.NewName(r.pos, "int16"), nil
	case KInt32:
		return syntax.NewName(r.pos, "int32"), nil
	case KInt64:
		return syntax.NewName(r.pos, "int64"), nil
	case KUint:
		return syntax.NewName(r.pos, "uint"), nil
	case KUint8:
		return syntax.NewName(r.pos, "uint8"), nil
	case KUint16:
		return syntax.NewName(r.pos, "uint16"), nil
	case KUint32:
		return syntax.NewName(r.pos, "uint32"), nil
	case KUint64:
		return syntax.NewName(r.pos, "uint64"), nil
	case KUintptr:
		return syntax.NewName(r.pos, "uintptr"), nil
	case KFloat32:
		return syntax.NewName(r.pos, "float32"), nil
	case KFloat64:
		return syntax.NewName(r.pos, "float64"), nil
	case KComplex64:
		return syntax.NewName(r.pos, "complex64"), nil
	case KComplex128:
		return syntax.NewName(r.pos, "complex128"), nil
	case KString:
		return syntax.NewName(r.pos, "string"), nil
	case KSlice:
		el, err := r.typeExpr(t.Elem)
		if err != nil {
			return nil, err
		}
		return &syntax.SliceType{Elem: el}, nil
	case KArray:
		el, err := r.typeExpr(t.Elem)
		if err != nil {
			return nil, err
		}
		return &syntax.ArrayType{Len: &syntax.BasicLit{Value: strconv.FormatInt(t.Len, 10), Kind: syntax.IntLit}, Elem: el}, nil
	case KMap:
		k, err := r.typeExpr(t.Key)
		if err != nil {
			return nil, err
		}
		v, err := r.typeExpr(t.Elem)
		if err != nil {
			return nil, err
		}
		return &syntax.MapType{Key: k, Value: v}, nil
	case KStruct:
		fields := make([]*syntax.Field, 0, len(t.Fields))
		for _, f := range t.Fields {
			ft, err := r.typeExpr(f.Type)
			if err != nil {
				return nil, err
			}
			fields = append(fields, &syntax.Field{Name: syntax.NewName(r.pos, f.Name), Type: ft})
		}
		return &syntax.StructType{FieldList: fields}, nil
	}
	return nil, fmt.Errorf("cannot spell type in const evaluation")
}
