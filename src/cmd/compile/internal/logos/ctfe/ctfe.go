// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ctfe

import (
	"cmd/compile/internal/syntax"
	"math/big"
)

type Config struct {
	MaxSteps      int
	MaxValueBytes int64
	WordSize      int64
}

func (c Config) withDefaults() Config {
	if c.MaxSteps == 0 {
		c.MaxSteps = 100000
	}
	if c.MaxValueBytes == 0 {
		c.MaxValueBytes = 1 << 20
	}
	if c.WordSize == 0 {
		c.WordSize = 8
	}
	return c
}

type ErrorCode int

const (
	CodeStepLimit ErrorCode = iota
	CodeTooLarge
	CodeEval
	CodeSilent
)

type Error struct {
	Code   ErrorCode
	Detail string
	Pos    syntax.Pos
}

func (e *Error) Error() string {
	switch e.Code {
	case CodeStepLimit:
		return "CTFE step limit exceeded"
	case CodeTooLarge:
		return "const value too large: " + e.Detail
	case CodeSilent:
		return "silent const abort: " + e.Detail
	}
	return e.Detail
}

type Kind int

const (
	KInvalid Kind = iota
	KBool
	KInt
	KInt8
	KInt16
	KInt32
	KInt64
	KUint
	KUint8
	KUint16
	KUint32
	KUint64
	KUintptr
	KFloat32
	KFloat64
	KString
	KSlice
	KArray
	KMap
	KStruct
	KUntypedInt
	KUntypedFloat
	KUntypedRune
	KUntypedString
	KUntypedBool
	KUntypedNil
	KComplex64
	KComplex128
	KUntypedComplex
)

type Type struct {
	Kind    Kind
	Len     int64
	Elem    *Type
	Key     *Type
	Fields  []Field
	Named   string
	PkgPath string
	Opaque  bool
	Impure  string
}

type Field struct {
	Name     string
	Type     *Type
	Embedded bool
}

func UntypedInt() *Type { return &Type{Kind: KUntypedInt} }

func UntypedFloat() *Type { return &Type{Kind: KUntypedFloat} }

func UntypedComplex() *Type { return &Type{Kind: KUntypedComplex} }

func UntypedRune() *Type { return &Type{Kind: KUntypedRune} }

func UntypedString() *Type { return &Type{Kind: KUntypedString} }

func UntypedBool() *Type { return &Type{Kind: KUntypedBool} }

func BasicType(name string) *Type {
	switch name {
	case "bool":
		return &Type{Kind: KBool}
	case "int":
		return &Type{Kind: KInt}
	case "int8":
		return &Type{Kind: KInt8}
	case "int16":
		return &Type{Kind: KInt16}
	case "int32":
		return &Type{Kind: KInt32}
	case "int64":
		return &Type{Kind: KInt64}
	case "uint":
		return &Type{Kind: KUint}
	case "uint8":
		return &Type{Kind: KUint8}
	case "uint16":
		return &Type{Kind: KUint16}
	case "uint32":
		return &Type{Kind: KUint32}
	case "uint64":
		return &Type{Kind: KUint64}
	case "uintptr":
		return &Type{Kind: KUintptr}
	case "float32":
		return &Type{Kind: KFloat32}
	case "float64":
		return &Type{Kind: KFloat64}
	case "complex64":
		return &Type{Kind: KComplex64}
	case "complex128":
		return &Type{Kind: KComplex128}
	case "string":
		return &Type{Kind: KString}
	}
	return nil
}

func intWidth(k Kind, word int64) int64 {
	switch k {
	case KInt8, KUint8:
		return 8
	case KInt16, KUint16:
		return 16
	case KInt32, KUint32, KFloat32:
		return 32
	case KInt64, KUint64, KFloat64, KComplex64:
		return 64
	case KComplex128:
		return 128
	case KInt, KUint, KUintptr:
		return word * 8
	}
	return 0
}

func isSigned(k Kind) bool {
	switch k {
	case KInt, KInt8, KInt16, KInt32, KInt64:
		return true
	}
	return false
}

func isInteger(k Kind) bool {
	switch k {
	case KInt, KInt8, KInt16, KInt32, KInt64,
		KUint, KUint8, KUint16, KUint32, KUint64, KUintptr,
		KUntypedInt, KUntypedRune:
		return true
	}
	return false
}

func isFloat(k Kind) bool {
	return k == KFloat32 || k == KFloat64 || k == KUntypedFloat
}

func isComplex(k Kind) bool {
	return k == KComplex64 || k == KComplex128 || k == KUntypedComplex
}

func isUntyped(k Kind) bool {
	switch k {
	case KUntypedInt, KUntypedFloat, KUntypedComplex, KUntypedRune, KUntypedString, KUntypedBool:
		return true
	}
	return false
}

func isStringLike(k Kind) bool {
	return k == KString || k == KUntypedString
}

func isNumeric(k Kind) bool {
	return isInteger(k) || isFloat(k) || isComplex(k)
}

func defaultType(k Kind) *Type {
	switch k {
	case KUntypedInt:
		return &Type{Kind: KInt}
	case KUntypedFloat:
		return &Type{Kind: KFloat64}
	case KUntypedComplex:
		return &Type{Kind: KComplex128}
	case KUntypedRune:
		return &Type{Kind: KInt32}
	case KUntypedString:
		return &Type{Kind: KString}
	case KUntypedBool:
		return &Type{Kind: KBool}
	}
	return &Type{Kind: KInvalid}
}

type sliceVal struct {
	backing *[]Value
	lo      int
	length  int
	cap_    int
}

type mapVal struct {
	keys []Value
	vals []Value
}

type Value struct {
	typ      *Type
	constant bool
	ival     *big.Int
	fval     *big.Rat
	f64      float64
	cval     complex128
	cparts   *[2]big.Rat
	sval     string
	bval     bool
	slice    *sliceVal
	array    []Value
	strct    []Value
	mval     *mapVal
	isNil    bool
}

func (v Value) Int64() int64 { return v.ival.Int64() }

func (v Value) BigInt() *big.Int {
	if v.ival == nil {
		return nil
	}
	return new(big.Int).Set(v.ival)
}

func (v Value) Bool() bool { return v.bval }

func (v Value) Str() string { return v.sval }

func (v Value) Float64() float64 {
	if v.fval != nil {
		f, _ := v.fval.Float64()
		return f
	}
	return v.f64
}

func (v Value) Complex() complex128 {
	if v.cparts != nil {
		re, _ := v.cparts[0].Float64()
		im, _ := v.cparts[1].Float64()
		return complex(re, im)
	}
	return v.cval
}

func (v Value) mapLen() int {
	if v.mval == nil {
		return 0
	}
	return len(v.mval.keys)
}

func (v Value) Fields() []Value { return v.strct }

func (v Value) Elems() []Value { return v.array }

func (v Value) MapEntries() (keys, vals []Value) {
	if v.mval == nil {
		return nil, nil
	}
	return v.mval.keys, v.mval.vals
}

func (v Value) mapKeyBits(i int) uint64 {
	return v.mval.keys[i].floatBits()
}

func (v Value) Type() *Type { return v.typ }

func (v Value) isConstant() bool { return v.constant }

func (v Value) IsValid() bool { return v.typ != nil }

func (v Value) IsComposite() bool {
	if v.typ == nil {
		return false
	}
	switch v.typ.Kind {
	case KSlice, KArray, KMap, KStruct:
		return true
	}
	return false
}

type EntityKind int

const (
	EntNone EntityKind = iota
	EntBuiltin
	EntConstFunc
	EntPlainFunc
	EntConst
	EntPackageVar
	EntImport
	EntType
	EntPackage
)

type Entity struct {
	Kind    EntityKind
	Builtin string
	Fn      *Func
	Const   Value
	Name    string
}

type Param struct {
	Decl *syntax.Name
	Type *Type
}

type Func struct {
	Name        string
	Params      []Param
	Variadic    bool
	Results     []*Type
	ResultNames []string
	Body        *syntax.BlockStmt
	Const       bool
	Generic     bool
}

type Host interface {
	Resolve(id *syntax.Name) Entity
	ResolveSel(sel *syntax.SelectorExpr) Entity
	TypeOf(e syntax.Expr) *Type
}

type Stats struct {
	Steps int
}

func Eval(h Host, cfg Config, e syntax.Expr) (Value, Stats, error) {
	ev := &evaluator{h: h, cfg: cfg.withDefaults()}
	top := ev.top()
	ev.frames = append(ev.frames, top)
	v, err := ev.evalExpr(top, e)
	if err != nil {
		return Value{}, Stats{Steps: ev.steps}, err
	}
	if _, err := ev.checkSize(v); err != nil {
		return Value{}, Stats{Steps: ev.steps}, err
	}
	return v, Stats{Steps: ev.steps}, nil
}
