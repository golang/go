// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ctfe

// This package implements the M5-S2 compile-time interpreter (SPEC Sec
// 2.1, T-201): a tree-walking evaluator over checked syntax for const
// composite initializers and calls to const funcs. Values cover
// integers with target widths, floats, strings, bools, arrays, slices,
// structs, and maps, plus control flow and calls to const funcs.
//
// The interpreter is deliberately independent of types2 (which would be
// an import cycle): all type information arrives through the Host
// interface, implemented by types2 on the checking side and by a fake
// in tests. Identifiers inside function bodies resolve through the
// interpreter's own block-structured scopes; only package-level and
// universe names reach the host.
//
// Sharing matches Go: slices alias their backing store (append
// reallocates exactly when the capacity is exhausted), maps are
// reference values, and arrays and structs copy on assignment. A const
// composite mutated through an alias still evaluates correctly,
// because lowering literalizes the final state and the lowered program
// re-executes the same pure calls in the same order.

import (
	"cmd/compile/internal/syntax"
	"math/big"
)

// Config controls evaluation limits (SPEC T-201, T-102).
type Config struct {
	// MaxSteps bounds evaluation; a run that exceeds it fails with
	// CodeStepLimit. Zero means DefaultConfig's value.
	MaxSteps int
	// MaxValueBytes caps the total distinct bytes of a const value;
	// a larger value fails with CodeTooLarge. Zero means default.
	MaxValueBytes int64
	// WordSize is the target width of int, uint, and uintptr in
	// bytes (4 or 8). Zero means 8.
	WordSize int64
}

// DefaultConfig returns the SPEC defaults: 100,000 steps, a 1 MB value
// cap, and 64-bit words. The word size is always overridden by the
// caller with the target width.
func DefaultConfig() Config {
	return Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
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

// ErrorCode classifies an evaluation failure.
type ErrorCode int

const (
	// CodeStepLimit reports [comptime::ctfe_step_limit].
	CodeStepLimit ErrorCode = iota
	// CodeTooLarge reports [comptime::const_too_large].
	CodeTooLarge
	// CodeEval reports [correctness::const_eval] with Detail.
	CodeEval
	// CodeSilent aborts without reporting: the impurity sits in a
	// const func body, whose static purity check reports it.
	CodeSilent
)

// Error is an evaluation failure at Pos with a Detail reason.
type Error struct {
	Code   ErrorCode
	Detail string
	Pos    syntax.Pos
	// silentIn marks the const function whose body holds the
	// impurity, for junctions that must not convert silence.
	silentIn *Func
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

// Kind describes the shape of a static type.
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
)

// Type is a compile-time type descriptor. Named carries a same-package
// defined type name for literal synthesis; Opaque marks types that
// cannot be spelled at the use site (lowering keeps the original call
// for those instead of a literal); Impure carries a purity-class reason
// for named types resolving behind pointers, channels, or uintptr.
type Type struct {
	Kind   Kind
	Len    int64 // arrays
	Elem   *Type // slice and array elements
	Key    *Type // map keys
	Fields []Field
	Named  string
	Opaque bool
	Impure string
}

// Field is one struct field.
type Field struct {
	Name     string
	Type     *Type
	Embedded bool
}

// UntypedInt returns the type of an untyped integer constant.
func UntypedInt() *Type { return &Type{Kind: KUntypedInt} }

// UntypedFloat returns the type of an untyped floating constant.
func UntypedFloat() *Type { return &Type{Kind: KUntypedFloat} }

// UntypedRune returns the type of an untyped rune constant.
func UntypedRune() *Type { return &Type{Kind: KUntypedRune} }

// UntypedString returns the type of an untyped string constant.
func UntypedString() *Type { return &Type{Kind: KUntypedString} }

// UntypedBool returns the type of an untyped boolean constant.
func UntypedBool() *Type { return &Type{Kind: KUntypedBool} }

// BasicType returns the descriptor for a predeclared basic type name.
// It returns nil for unknown names.
func BasicType(name string, word int64) *Type {
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
	case "string":
		return &Type{Kind: KString}
	}
	return nil
}

// intWidth returns the width in bits of an integer kind.
func intWidth(k Kind, word int64) int64 {
	switch k {
	case KInt8, KUint8:
		return 8
	case KInt16, KUint16:
		return 16
	case KInt32, KUint32, KFloat32:
		return 32
	case KInt64, KUint64, KFloat64:
		return 64
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

func isUntyped(k Kind) bool {
	switch k {
	case KUntypedInt, KUntypedFloat, KUntypedRune, KUntypedString, KUntypedBool:
		return true
	}
	return false
}

func isStringLike(k Kind) bool {
	return k == KString || k == KUntypedString
}

func isNumeric(k Kind) bool {
	return isInteger(k) || isFloat(k)
}

// defaultType maps an untyped kind to its default type.
func defaultType(k Kind, word int64) *Type {
	switch k {
	case KUntypedInt:
		return &Type{Kind: KInt}
	case KUntypedFloat:
		return &Type{Kind: KFloat64}
	case KUntypedRune:
		return &Type{Kind: KInt32}
	case KUntypedString:
		return &Type{Kind: KString}
	case KUntypedBool:
		return &Type{Kind: KBool}
	}
	return &Type{Kind: KInvalid}
}

// sliceVal is a shared slice header: backing is aliased across copies
// until append exhausts the capacity, exactly like Go.
type sliceVal struct {
	backing *[]Value
	lo      int
	length  int
	cap_    int
}

// mapVal is a reference map value preserving insertion order.
type mapVal struct {
	keys []Value
	vals []Value
}

// Value is one compile-time value. The constant flag selects Go
// constant semantics (exact arithmetic, range-checked conversions) for
// literal- and const-derived values, and variable semantics (wrapping)
// for parameters, locals, and values loaded out of composites.
type Value struct {
	typ      *Type
	constant bool
	ival     *big.Int
	fval     *big.Rat
	f64      float64
	sval     string
	bval     bool
	slice    *sliceVal
	array    []Value
	strct    []Value
	mval     *mapVal
	isNil    bool
}

// Int64 returns the integer payload (tests and rendering).
func (v Value) Int64() int64 { return v.ival.Int64() }

// BigInt returns the exact integer payload; callers must not mutate it.
func (v Value) BigInt() *big.Int { return v.ival }

// Bool returns the boolean payload.
func (v Value) Bool() bool { return v.bval }

// Str returns the string payload.
func (v Value) Str() string { return v.sval }

// Float64 returns the float payload in 64-bit form.
func (v Value) Float64() float64 {
	if v.fval != nil {
		f, _ := v.fval.Float64()
		return f
	}
	return v.f64
}

// MapLen returns the number of map entries.
func (v Value) MapLen() int {
	if v.mval == nil {
		return 0
	}
	return len(v.mval.keys)
}

// Fields returns the field values of a struct value in declaration order.
func (v Value) Fields() []Value { return v.strct }

// Elems returns the element values of an array value in index order.
func (v Value) Elems() []Value { return v.array }

// MapEntries returns the keys and values of a map value in insertion order.
func (v Value) MapEntries() (keys, vals []Value) {
	if v.mval == nil {
		return nil, nil
	}
	return v.mval.keys, v.mval.vals
}

// MapKeyBits returns the float64 bits of map key i (tests).
func (v Value) MapKeyBits(i int) uint64 {
	return v.mval.keys[i].floatBits()
}

// Type returns the value's descriptor.
func (v Value) Type() *Type { return v.typ }

// IsConstant reports constant semantics.
func (v Value) IsConstant() bool { return v.constant }

// IsValid reports whether the value holds a computed result (as
// opposed to a memoized failure marker).
func (v Value) IsValid() bool { return v.typ != nil }

// EntityKind classifies a resolved identifier.
type EntityKind int

const (
	// EntNone resolves to nothing usable (universe constants like
	// true/nil, or unknown names the checker already rejected).
	EntNone EntityKind = iota
	// EntBuiltin is a builtin function call.
	EntBuiltin
	// EntConstFunc is a call to a const func.
	EntConstFunc
	// EntPlainFunc is a call to a same-package plain func, evaluated
	// transitively after its own purity scan.
	EntPlainFunc
	// EntConst is a compile-time known value (constant or recorded
	// const composite).
	EntConst
	// EntPackageVar is a package-level variable: reading or writing
	// one is impure.
	EntPackageVar
	// EntImport is an imported function, variable, or package use.
	EntImport
	// EntType is a type name in value position.
	EntType
	// EntPackage is a package name: selectors on it resolve through
	// ResolveSel.
	EntPackage
)

// Entity is one resolved identifier.
type Entity struct {
	Kind    EntityKind
	Builtin string
	Fn      *Func
	Const   Value
	Name    string
	Type    *Type
}

// Param is one function parameter or result.
type Param struct {
	Decl *syntax.Name
	Type *Type
}

// Func is one callable function body. ResultNames parallels Results
// with "" for unnamed results. Generic marks an uninstantiated generic
// function, which const evaluation refuses.
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

// Host abstracts the type checker for the interpreter.
type Host interface {
	// Resolve maps an identifier to its entity.
	Resolve(id *syntax.Name) Entity
	// ResolveSel maps a selector to its entity (qualified
	// references and method calls).
	ResolveSel(sel *syntax.SelectorExpr) Entity
	// TypeOf returns the static type of e: composite literal types,
	// conversion targets, declared variable types, and make operands.
	TypeOf(e syntax.Expr) *Type
	// WordSize is the target int/uint width in bytes.
	WordSize() int64
}

// Stats carries evaluation measurements.
type Stats struct {
	Steps int
}

// Eval evaluates a single-valued const initializer.
func Eval(h Host, cfg Config, e syntax.Expr) (Value, Stats, error) {
	ev := &evaluator{h: h, cfg: cfg.withDefaults()}
	top := ev.top()
	ev.frames = append(ev.frames, top)
	v, err := ev.evalExpr(top, e)
	if err != nil {
		return Value{}, Stats{Steps: ev.steps}, err
	}
	if err := ev.checkSize(v); err != nil {
		return Value{}, Stats{Steps: ev.steps}, err
	}
	return v, Stats{Steps: ev.steps}, nil
}
