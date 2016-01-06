// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !go1.5

package interp

import (
	"bytes"
	"fmt"
	"go/token"
	"strings"
	"sync"
	"unsafe"

	"golang.org/x/tools/go/exact"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/types"
)

// If the target program panics, the interpreter panics with this type.
type targetPanic struct {
	v value
}

func (p targetPanic) String() string {
	return toString(p.v)
}

// If the target program calls exit, the interpreter panics with this type.
type exitPanic int

// constValue returns the value of the constant with the
// dynamic type tag appropriate for c.Type().
func constValue(c *ssa.Const) value {
	if c.IsNil() {
		return zero(c.Type()) // typed nil
	}

	if t, ok := c.Type().Underlying().(*types.Basic); ok {
		// TODO(adonovan): eliminate untyped constants from SSA form.
		switch t.Kind() {
		case types.Bool, types.UntypedBool:
			return exact.BoolVal(c.Value)
		case types.Int, types.UntypedInt:
			// Assume sizeof(int) is same on host and target.
			return int(c.Int64())
		case types.Int8:
			return int8(c.Int64())
		case types.Int16:
			return int16(c.Int64())
		case types.Int32, types.UntypedRune:
			return int32(c.Int64())
		case types.Int64:
			return c.Int64()
		case types.Uint:
			// Assume sizeof(uint) is same on host and target.
			return uint(c.Uint64())
		case types.Uint8:
			return uint8(c.Uint64())
		case types.Uint16:
			return uint16(c.Uint64())
		case types.Uint32:
			return uint32(c.Uint64())
		case types.Uint64:
			return c.Uint64()
		case types.Uintptr:
			// Assume sizeof(uintptr) is same on host and target.
			return uintptr(c.Uint64())
		case types.Float32:
			return float32(c.Float64())
		case types.Float64, types.UntypedFloat:
			return c.Float64()
		case types.Complex64:
			return complex64(c.Complex128())
		case types.Complex128, types.UntypedComplex:
			return c.Complex128()
		case types.String, types.UntypedString:
			if c.Value.Kind() == exact.String {
				return exact.StringVal(c.Value)
			}
			return string(rune(c.Int64()))
		}
	}

	panic(fmt.Sprintf("constValue: %s", c))
}

// asInt converts x, which must be an integer, to an int suitable for
// use as a slice or array index or operand to make().
func asInt(x value) int {
	switch x := x.(type) {
	case int:
		return x
	case int8:
		return int(x)
	case int16:
		return int(x)
	case int32:
		return int(x)
	case int64:
		return int(x)
	case uint:
		return int(x)
	case uint8:
		return int(x)
	case uint16:
		return int(x)
	case uint32:
		return int(x)
	case uint64:
		return int(x)
	case uintptr:
		return int(x)
	}
	panic(fmt.Sprintf("cannot convert %T to int", x))
}

// asUint64 converts x, which must be an unsigned integer, to a uint64
// suitable for use as a bitwise shift count.
func asUint64(x value) uint64 {
	switch x := x.(type) {
	case uint:
		return uint64(x)
	case uint8:
		return uint64(x)
	case uint16:
		return uint64(x)
	case uint32:
		return uint64(x)
	case uint64:
		return x
	case uintptr:
		return uint64(x)
	}
	panic(fmt.Sprintf("cannot convert %T to uint64", x))
}

// zero returns a new "zero" value of the specified type.
func zero(t types.Type) value {
	switch t := t.(type) {
	case *types.Basic:
		if t.Kind() == types.UntypedNil {
			panic("untyped nil has no zero value")
		}
		if t.Info()&types.IsUntyped != 0 {
			// TODO(adonovan): make it an invariant that
			// this is unreachable.  Currently some
			// constants have 'untyped' types when they
			// should be defaulted by the typechecker.
			t = ssa.DefaultType(t).(*types.Basic)
		}
		switch t.Kind() {
		case types.Bool:
			return false
		case types.Int:
			return int(0)
		case types.Int8:
			return int8(0)
		case types.Int16:
			return int16(0)
		case types.Int32:
			return int32(0)
		case types.Int64:
			return int64(0)
		case types.Uint:
			return uint(0)
		case types.Uint8:
			return uint8(0)
		case types.Uint16:
			return uint16(0)
		case types.Uint32:
			return uint32(0)
		case types.Uint64:
			return uint64(0)
		case types.Uintptr:
			return uintptr(0)
		case types.Float32:
			return float32(0)
		case types.Float64:
			return float64(0)
		case types.Complex64:
			return complex64(0)
		case types.Complex128:
			return complex128(0)
		case types.String:
			return ""
		case types.UnsafePointer:
			return unsafe.Pointer(nil)
		default:
			panic(fmt.Sprint("zero for unexpected type:", t))
		}
	case *types.Pointer:
		return (*value)(nil)
	case *types.Array:
		a := make(array, t.Len())
		for i := range a {
			a[i] = zero(t.Elem())
		}
		return a
	case *types.Named:
		return zero(t.Underlying())
	case *types.Interface:
		return iface{} // nil type, methodset and value
	case *types.Slice:
		return []value(nil)
	case *types.Struct:
		s := make(structure, t.NumFields())
		for i := range s {
			s[i] = zero(t.Field(i).Type())
		}
		return s
	case *types.Tuple:
		if t.Len() == 1 {
			return zero(t.At(0).Type())
		}
		s := make(tuple, t.Len())
		for i := range s {
			s[i] = zero(t.At(i).Type())
		}
		return s
	case *types.Chan:
		return chan value(nil)
	case *types.Map:
		if usesBuiltinMap(t.Key()) {
			return map[value]value(nil)
		}
		return (*hashmap)(nil)
	case *types.Signature:
		return (*ssa.Function)(nil)
	}
	panic(fmt.Sprint("zero: unexpected ", t))
}

// slice returns x[lo:hi:max].  Any of lo, hi and max may be nil.
func slice(x, lo, hi, max value) value {
	var Len, Cap int
	switch x := x.(type) {
	case string:
		Len = len(x)
	case []value:
		Len = len(x)
		Cap = cap(x)
	case *value: // *array
		a := (*x).(array)
		Len = len(a)
		Cap = cap(a)
	}

	l := 0
	if lo != nil {
		l = asInt(lo)
	}

	h := Len
	if hi != nil {
		h = asInt(hi)
	}

	m := Cap
	if max != nil {
		m = asInt(max)
	}

	switch x := x.(type) {
	case string:
		return x[l:h]
	case []value:
		return x[l:h:m]
	case *value: // *array
		a := (*x).(array)
		return []value(a)[l:h:m]
	}
	panic(fmt.Sprintf("slice: unexpected X type: %T", x))
}

// lookup returns x[idx] where x is a map or string.
func lookup(instr *ssa.Lookup, x, idx value) value {
	switch x := x.(type) { // map or string
	case map[value]value, *hashmap:
		var v value
		var ok bool
		switch x := x.(type) {
		case map[value]value:
			v, ok = x[idx]
		case *hashmap:
			v = x.lookup(idx.(hashable))
			ok = v != nil
		}
		if !ok {
			v = zero(instr.X.Type().Underlying().(*types.Map).Elem())
		}
		if instr.CommaOk {
			v = tuple{v, ok}
		}
		return v
	case string:
		return x[asInt(idx)]
	}
	panic(fmt.Sprintf("unexpected x type in Lookup: %T", x))
}

// binop implements all arithmetic and logical binary operators for
// numeric datatypes and strings.  Both operands must have identical
// dynamic type.
//
func binop(op token.Token, t types.Type, x, y value) value {
	switch op {
	case token.ADD:
		switch x.(type) {
		case int:
			return x.(int) + y.(int)
		case int8:
			return x.(int8) + y.(int8)
		case int16:
			return x.(int16) + y.(int16)
		case int32:
			return x.(int32) + y.(int32)
		case int64:
			return x.(int64) + y.(int64)
		case uint:
			return x.(uint) + y.(uint)
		case uint8:
			return x.(uint8) + y.(uint8)
		case uint16:
			return x.(uint16) + y.(uint16)
		case uint32:
			return x.(uint32) + y.(uint32)
		case uint64:
			return x.(uint64) + y.(uint64)
		case uintptr:
			return x.(uintptr) + y.(uintptr)
		case float32:
			return x.(float32) + y.(float32)
		case float64:
			return x.(float64) + y.(float64)
		case complex64:
			return x.(complex64) + y.(complex64)
		case complex128:
			return x.(complex128) + y.(complex128)
		case string:
			return x.(string) + y.(string)
		}

	case token.SUB:
		switch x.(type) {
		case int:
			return x.(int) - y.(int)
		case int8:
			return x.(int8) - y.(int8)
		case int16:
			return x.(int16) - y.(int16)
		case int32:
			return x.(int32) - y.(int32)
		case int64:
			return x.(int64) - y.(int64)
		case uint:
			return x.(uint) - y.(uint)
		case uint8:
			return x.(uint8) - y.(uint8)
		case uint16:
			return x.(uint16) - y.(uint16)
		case uint32:
			return x.(uint32) - y.(uint32)
		case uint64:
			return x.(uint64) - y.(uint64)
		case uintptr:
			return x.(uintptr) - y.(uintptr)
		case float32:
			return x.(float32) - y.(float32)
		case float64:
			return x.(float64) - y.(float64)
		case complex64:
			return x.(complex64) - y.(complex64)
		case complex128:
			return x.(complex128) - y.(complex128)
		}

	case token.MUL:
		switch x.(type) {
		case int:
			return x.(int) * y.(int)
		case int8:
			return x.(int8) * y.(int8)
		case int16:
			return x.(int16) * y.(int16)
		case int32:
			return x.(int32) * y.(int32)
		case int64:
			return x.(int64) * y.(int64)
		case uint:
			return x.(uint) * y.(uint)
		case uint8:
			return x.(uint8) * y.(uint8)
		case uint16:
			return x.(uint16) * y.(uint16)
		case uint32:
			return x.(uint32) * y.(uint32)
		case uint64:
			return x.(uint64) * y.(uint64)
		case uintptr:
			return x.(uintptr) * y.(uintptr)
		case float32:
			return x.(float32) * y.(float32)
		case float64:
			return x.(float64) * y.(float64)
		case complex64:
			return x.(complex64) * y.(complex64)
		case complex128:
			return x.(complex128) * y.(complex128)
		}

	case token.QUO:
		switch x.(type) {
		case int:
			return x.(int) / y.(int)
		case int8:
			return x.(int8) / y.(int8)
		case int16:
			return x.(int16) / y.(int16)
		case int32:
			return x.(int32) / y.(int32)
		case int64:
			return x.(int64) / y.(int64)
		case uint:
			return x.(uint) / y.(uint)
		case uint8:
			return x.(uint8) / y.(uint8)
		case uint16:
			return x.(uint16) / y.(uint16)
		case uint32:
			return x.(uint32) / y.(uint32)
		case uint64:
			return x.(uint64) / y.(uint64)
		case uintptr:
			return x.(uintptr) / y.(uintptr)
		case float32:
			return x.(float32) / y.(float32)
		case float64:
			return x.(float64) / y.(float64)
		case complex64:
			return x.(complex64) / y.(complex64)
		case complex128:
			return x.(complex128) / y.(complex128)
		}

	case token.REM:
		switch x.(type) {
		case int:
			return x.(int) % y.(int)
		case int8:
			return x.(int8) % y.(int8)
		case int16:
			return x.(int16) % y.(int16)
		case int32:
			return x.(int32) % y.(int32)
		case int64:
			return x.(int64) % y.(int64)
		case uint:
			return x.(uint) % y.(uint)
		case uint8:
			return x.(uint8) % y.(uint8)
		case uint16:
			return x.(uint16) % y.(uint16)
		case uint32:
			return x.(uint32) % y.(uint32)
		case uint64:
			return x.(uint64) % y.(uint64)
		case uintptr:
			return x.(uintptr) % y.(uintptr)
		}

	case token.AND:
		switch x.(type) {
		case int:
			return x.(int) & y.(int)
		case int8:
			return x.(int8) & y.(int8)
		case int16:
			return x.(int16) & y.(int16)
		case int32:
			return x.(int32) & y.(int32)
		case int64:
			return x.(int64) & y.(int64)
		case uint:
			return x.(uint) & y.(uint)
		case uint8:
			return x.(uint8) & y.(uint8)
		case uint16:
			return x.(uint16) & y.(uint16)
		case uint32:
			return x.(uint32) & y.(uint32)
		case uint64:
			return x.(uint64) & y.(uint64)
		case uintptr:
			return x.(uintptr) & y.(uintptr)
		}

	case token.OR:
		switch x.(type) {
		case int:
			return x.(int) | y.(int)
		case int8:
			return x.(int8) | y.(int8)
		case int16:
			return x.(int16) | y.(int16)
		case int32:
			return x.(int32) | y.(int32)
		case int64:
			return x.(int64) | y.(int64)
		case uint:
			return x.(uint) | y.(uint)
		case uint8:
			return x.(uint8) | y.(uint8)
		case uint16:
			return x.(uint16) | y.(uint16)
		case uint32:
			return x.(uint32) | y.(uint32)
		case uint64:
			return x.(uint64) | y.(uint64)
		case uintptr:
			return x.(uintptr) | y.(uintptr)
		}

	case token.XOR:
		switch x.(type) {
		case int:
			return x.(int) ^ y.(int)
		case int8:
			return x.(int8) ^ y.(int8)
		case int16:
			return x.(int16) ^ y.(int16)
		case int32:
			return x.(int32) ^ y.(int32)
		case int64:
			return x.(int64) ^ y.(int64)
		case uint:
			return x.(uint) ^ y.(uint)
		case uint8:
			return x.(uint8) ^ y.(uint8)
		case uint16:
			return x.(uint16) ^ y.(uint16)
		case uint32:
			return x.(uint32) ^ y.(uint32)
		case uint64:
			return x.(uint64) ^ y.(uint64)
		case uintptr:
			return x.(uintptr) ^ y.(uintptr)
		}

	case token.AND_NOT:
		switch x.(type) {
		case int:
			return x.(int) &^ y.(int)
		case int8:
			return x.(int8) &^ y.(int8)
		case int16:
			return x.(int16) &^ y.(int16)
		case int32:
			return x.(int32) &^ y.(int32)
		case int64:
			return x.(int64) &^ y.(int64)
		case uint:
			return x.(uint) &^ y.(uint)
		case uint8:
			return x.(uint8) &^ y.(uint8)
		case uint16:
			return x.(uint16) &^ y.(uint16)
		case uint32:
			return x.(uint32) &^ y.(uint32)
		case uint64:
			return x.(uint64) &^ y.(uint64)
		case uintptr:
			return x.(uintptr) &^ y.(uintptr)
		}

	case token.SHL:
		y := asUint64(y)
		switch x.(type) {
		case int:
			return x.(int) << y
		case int8:
			return x.(int8) << y
		case int16:
			return x.(int16) << y
		case int32:
			return x.(int32) << y
		case int64:
			return x.(int64) << y
		case uint:
			return x.(uint) << y
		case uint8:
			return x.(uint8) << y
		case uint16:
			return x.(uint16) << y
		case uint32:
			return x.(uint32) << y
		case uint64:
			return x.(uint64) << y
		case uintptr:
			return x.(uintptr) << y
		}

	case token.SHR:
		y := asUint64(y)
		switch x.(type) {
		case int:
			return x.(int) >> y
		case int8:
			return x.(int8) >> y
		case int16:
			return x.(int16) >> y
		case int32:
			return x.(int32) >> y
		case int64:
			return x.(int64) >> y
		case uint:
			return x.(uint) >> y
		case uint8:
			return x.(uint8) >> y
		case uint16:
			return x.(uint16) >> y
		case uint32:
			return x.(uint32) >> y
		case uint64:
			return x.(uint64) >> y
		case uintptr:
			return x.(uintptr) >> y
		}

	case token.LSS:
		switch x.(type) {
		case int:
			return x.(int) < y.(int)
		case int8:
			return x.(int8) < y.(int8)
		case int16:
			return x.(int16) < y.(int16)
		case int32:
			return x.(int32) < y.(int32)
		case int64:
			return x.(int64) < y.(int64)
		case uint:
			return x.(uint) < y.(uint)
		case uint8:
			return x.(uint8) < y.(uint8)
		case uint16:
			return x.(uint16) < y.(uint16)
		case uint32:
			return x.(uint32) < y.(uint32)
		case uint64:
			return x.(uint64) < y.(uint64)
		case uintptr:
			return x.(uintptr) < y.(uintptr)
		case float32:
			return x.(float32) < y.(float32)
		case float64:
			return x.(float64) < y.(float64)
		case string:
			return x.(string) < y.(string)
		}

	case token.LEQ:
		switch x.(type) {
		case int:
			return x.(int) <= y.(int)
		case int8:
			return x.(int8) <= y.(int8)
		case int16:
			return x.(int16) <= y.(int16)
		case int32:
			return x.(int32) <= y.(int32)
		case int64:
			return x.(int64) <= y.(int64)
		case uint:
			return x.(uint) <= y.(uint)
		case uint8:
			return x.(uint8) <= y.(uint8)
		case uint16:
			return x.(uint16) <= y.(uint16)
		case uint32:
			return x.(uint32) <= y.(uint32)
		case uint64:
			return x.(uint64) <= y.(uint64)
		case uintptr:
			return x.(uintptr) <= y.(uintptr)
		case float32:
			return x.(float32) <= y.(float32)
		case float64:
			return x.(float64) <= y.(float64)
		case string:
			return x.(string) <= y.(string)
		}

	case token.EQL:
		return eqnil(t, x, y)

	case token.NEQ:
		return !eqnil(t, x, y)

	case token.GTR:
		switch x.(type) {
		case int:
			return x.(int) > y.(int)
		case int8:
			return x.(int8) > y.(int8)
		case int16:
			return x.(int16) > y.(int16)
		case int32:
			return x.(int32) > y.(int32)
		case int64:
			return x.(int64) > y.(int64)
		case uint:
			return x.(uint) > y.(uint)
		case uint8:
			return x.(uint8) > y.(uint8)
		case uint16:
			return x.(uint16) > y.(uint16)
		case uint32:
			return x.(uint32) > y.(uint32)
		case uint64:
			return x.(uint64) > y.(uint64)
		case uintptr:
			return x.(uintptr) > y.(uintptr)
		case float32:
			return x.(float32) > y.(float32)
		case float64:
			return x.(float64) > y.(float64)
		case string:
			return x.(string) > y.(string)
		}

	case token.GEQ:
		switch x.(type) {
		case int:
			return x.(int) >= y.(int)
		case int8:
			return x.(int8) >= y.(int8)
		case int16:
			return x.(int16) >= y.(int16)
		case int32:
			return x.(int32) >= y.(int32)
		case int64:
			return x.(int64) >= y.(int64)
		case uint:
			return x.(uint) >= y.(uint)
		case uint8:
			return x.(uint8) >= y.(uint8)
		case uint16:
			return x.(uint16) >= y.(uint16)
		case uint32:
			return x.(uint32) >= y.(uint32)
		case uint64:
			return x.(uint64) >= y.(uint64)
		case uintptr:
			return x.(uintptr) >= y.(uintptr)
		case float32:
			return x.(float32) >= y.(float32)
		case float64:
			return x.(float64) >= y.(float64)
		case string:
			return x.(string) >= y.(string)
		}
	}
	panic(fmt.Sprintf("invalid binary op: %T %s %T", x, op, y))
}

// eqnil returns the comparison x == y using the equivalence relation
// appropriate for type t.
// If t is a reference type, at most one of x or y may be a nil value
// of that type.
//
func eqnil(t types.Type, x, y value) bool {
	switch t.Underlying().(type) {
	case *types.Map, *types.Signature, *types.Slice:
		// Since these types don't support comparison,
		// one of the operands must be a literal nil.
		switch x := x.(type) {
		case *hashmap:
			return (x != nil) == (y.(*hashmap) != nil)
		case map[value]value:
			return (x != nil) == (y.(map[value]value) != nil)
		case *ssa.Function:
			switch y := y.(type) {
			case *ssa.Function:
				return (x != nil) == (y != nil)
			case *closure:
				return true
			}
		case *closure:
			return (x != nil) == (y.(*ssa.Function) != nil)
		case []value:
			return (x != nil) == (y.([]value) != nil)
		}
		panic(fmt.Sprintf("eqnil(%s): illegal dynamic type: %T", t, x))
	}

	return equals(t, x, y)
}

func unop(instr *ssa.UnOp, x value) value {
	switch instr.Op {
	case token.ARROW: // receive
		v, ok := <-x.(chan value)
		if !ok {
			v = zero(instr.X.Type().Underlying().(*types.Chan).Elem())
		}
		if instr.CommaOk {
			v = tuple{v, ok}
		}
		return v
	case token.SUB:
		switch x := x.(type) {
		case int:
			return -x
		case int8:
			return -x
		case int16:
			return -x
		case int32:
			return -x
		case int64:
			return -x
		case uint:
			return -x
		case uint8:
			return -x
		case uint16:
			return -x
		case uint32:
			return -x
		case uint64:
			return -x
		case uintptr:
			return -x
		case float32:
			return -x
		case float64:
			return -x
		case complex64:
			return -x
		case complex128:
			return -x
		}
	case token.MUL:
		return load(deref(instr.X.Type()), x.(*value))
	case token.NOT:
		return !x.(bool)
	case token.XOR:
		switch x := x.(type) {
		case int:
			return ^x
		case int8:
			return ^x
		case int16:
			return ^x
		case int32:
			return ^x
		case int64:
			return ^x
		case uint:
			return ^x
		case uint8:
			return ^x
		case uint16:
			return ^x
		case uint32:
			return ^x
		case uint64:
			return ^x
		case uintptr:
			return ^x
		}
	}
	panic(fmt.Sprintf("invalid unary op %s %T", instr.Op, x))
}

// typeAssert checks whether dynamic type of itf is instr.AssertedType.
// It returns the extracted value on success, and panics on failure,
// unless instr.CommaOk, in which case it always returns a "value,ok" tuple.
//
func typeAssert(i *interpreter, instr *ssa.TypeAssert, itf iface) value {
	var v value
	err := ""
	if itf.t == nil {
		err = fmt.Sprintf("interface conversion: interface is nil, not %s", instr.AssertedType)

	} else if idst, ok := instr.AssertedType.Underlying().(*types.Interface); ok {
		v = itf
		err = checkInterface(i, idst, itf)

	} else if types.Identical(itf.t, instr.AssertedType) {
		v = itf.v // extract value

	} else {
		err = fmt.Sprintf("interface conversion: interface is %s, not %s", itf.t, instr.AssertedType)
	}

	if err != "" {
		if !instr.CommaOk {
			panic(err)
		}
		return tuple{zero(instr.AssertedType), false}
	}
	if instr.CommaOk {
		return tuple{v, true}
	}
	return v
}

// If CapturedOutput is non-nil, all writes by the interpreted program
// to file descriptors 1 and 2 will also be written to CapturedOutput.
//
// (The $GOROOT/test system requires that the test be considered a
// failure if "BUG" appears in the combined stdout/stderr output, even
// if it exits zero.  This is a global variable shared by all
// interpreters in the same process.)
//
var CapturedOutput *bytes.Buffer
var capturedOutputMu sync.Mutex

// write writes bytes b to the target program's file descriptor fd.
// The print/println built-ins and the write() system call funnel
// through here so they can be captured by the test driver.
func write(fd int, b []byte) (int, error) {
	// TODO(adonovan): fix: on Windows, std{out,err} are not 1, 2.
	if CapturedOutput != nil && (fd == 1 || fd == 2) {
		capturedOutputMu.Lock()
		CapturedOutput.Write(b) // ignore errors
		capturedOutputMu.Unlock()
	}
	return syswrite(fd, b)
}

// callBuiltin interprets a call to builtin fn with arguments args,
// returning its result.
func callBuiltin(caller *frame, callpos token.Pos, fn *ssa.Builtin, args []value) value {
	switch fn.Name() {
	case "append":
		if len(args) == 1 {
			return args[0]
		}
		if s, ok := args[1].(string); ok {
			// append([]byte, ...string) []byte
			arg0 := args[0].([]value)
			for i := 0; i < len(s); i++ {
				arg0 = append(arg0, s[i])
			}
			return arg0
		}
		// append([]T, ...[]T) []T
		return append(args[0].([]value), args[1].([]value)...)

	case "copy": // copy([]T, []T) int or copy([]byte, string) int
		src := args[1]
		if _, ok := src.(string); ok {
			params := fn.Type().(*types.Signature).Params()
			src = conv(params.At(0).Type(), params.At(1).Type(), src)
		}
		return copy(args[0].([]value), src.([]value))

	case "close": // close(chan T)
		close(args[0].(chan value))
		return nil

	case "delete": // delete(map[K]value, K)
		switch m := args[0].(type) {
		case map[value]value:
			delete(m, args[1])
		case *hashmap:
			m.delete(args[1].(hashable))
		default:
			panic(fmt.Sprintf("illegal map type: %T", m))
		}
		return nil

	case "print", "println": // print(any, ...)
		ln := fn.Name() == "println"
		var buf bytes.Buffer
		for i, arg := range args {
			if i > 0 && ln {
				buf.WriteRune(' ')
			}
			buf.WriteString(toString(arg))
		}
		if ln {
			buf.WriteRune('\n')
		}
		write(1, buf.Bytes())
		return nil

	case "len":
		switch x := args[0].(type) {
		case string:
			return len(x)
		case array:
			return len(x)
		case *value:
			return len((*x).(array))
		case []value:
			return len(x)
		case map[value]value:
			return len(x)
		case *hashmap:
			return x.len()
		case chan value:
			return len(x)
		default:
			panic(fmt.Sprintf("len: illegal operand: %T", x))
		}

	case "cap":
		switch x := args[0].(type) {
		case array:
			return cap(x)
		case *value:
			return cap((*x).(array))
		case []value:
			return cap(x)
		case chan value:
			return cap(x)
		default:
			panic(fmt.Sprintf("cap: illegal operand: %T", x))
		}

	case "real":
		switch c := args[0].(type) {
		case complex64:
			return real(c)
		case complex128:
			return real(c)
		default:
			panic(fmt.Sprintf("real: illegal operand: %T", c))
		}

	case "imag":
		switch c := args[0].(type) {
		case complex64:
			return imag(c)
		case complex128:
			return imag(c)
		default:
			panic(fmt.Sprintf("imag: illegal operand: %T", c))
		}

	case "complex":
		switch f := args[0].(type) {
		case float32:
			return complex(f, args[1].(float32))
		case float64:
			return complex(f, args[1].(float64))
		default:
			panic(fmt.Sprintf("complex: illegal operand: %T", f))
		}

	case "panic":
		// ssa.Panic handles most cases; this is only for "go
		// panic" or "defer panic".
		panic(targetPanic{args[0]})

	case "recover":
		return doRecover(caller)

	case "ssa:wrapnilchk":
		recv := args[0]
		if recv.(*value) == nil {
			recvType := args[1]
			methodName := args[2]
			panic(fmt.Sprintf("value method (%s).%s called using nil *%s pointer",
				recvType, methodName, recvType))
		}
		return recv
	}

	panic("unknown built-in: " + fn.Name())
}

func rangeIter(x value, t types.Type) iter {
	switch x := x.(type) {
	case map[value]value:
		// TODO(adonovan): fix: leaks goroutines and channels
		// on each incomplete map iteration.  We need to open
		// up an iteration interface using the
		// reflect.(Value).MapKeys machinery.
		it := make(mapIter)
		go func() {
			for k, v := range x {
				it <- [2]value{k, v}
			}
			close(it)
		}()
		return it
	case *hashmap:
		// TODO(adonovan): fix: leaks goroutines and channels
		// on each incomplete map iteration.  We need to open
		// up an iteration interface using the
		// reflect.(Value).MapKeys machinery.
		it := make(mapIter)
		go func() {
			for _, e := range x.table {
				for e != nil {
					it <- [2]value{e.key, e.value}
					e = e.next
				}
			}
			close(it)
		}()
		return it
	case string:
		return &stringIter{Reader: strings.NewReader(x)}
	}
	panic(fmt.Sprintf("cannot range over %T", x))
}

// widen widens a basic typed value x to the widest type of its
// category, one of:
//   bool, int64, uint64, float64, complex128, string.
// This is inefficient but reduces the size of the cross-product of
// cases we have to consider.
//
func widen(x value) value {
	switch y := x.(type) {
	case bool, int64, uint64, float64, complex128, string, unsafe.Pointer:
		return x
	case int:
		return int64(y)
	case int8:
		return int64(y)
	case int16:
		return int64(y)
	case int32:
		return int64(y)
	case uint:
		return uint64(y)
	case uint8:
		return uint64(y)
	case uint16:
		return uint64(y)
	case uint32:
		return uint64(y)
	case uintptr:
		return uint64(y)
	case float32:
		return float64(y)
	case complex64:
		return complex128(y)
	}
	panic(fmt.Sprintf("cannot widen %T", x))
}

// conv converts the value x of type t_src to type t_dst and returns
// the result.
// Possible cases are described with the ssa.Convert operator.
//
func conv(t_dst, t_src types.Type, x value) value {
	ut_src := t_src.Underlying()
	ut_dst := t_dst.Underlying()

	// Destination type is not an "untyped" type.
	if b, ok := ut_dst.(*types.Basic); ok && b.Info()&types.IsUntyped != 0 {
		panic("oops: conversion to 'untyped' type: " + b.String())
	}

	// Nor is it an interface type.
	if _, ok := ut_dst.(*types.Interface); ok {
		if _, ok := ut_src.(*types.Interface); ok {
			panic("oops: Convert should be ChangeInterface")
		} else {
			panic("oops: Convert should be MakeInterface")
		}
	}

	// Remaining conversions:
	//    + untyped string/number/bool constant to a specific
	//      representation.
	//    + conversions between non-complex numeric types.
	//    + conversions between complex numeric types.
	//    + integer/[]byte/[]rune -> string.
	//    + string -> []byte/[]rune.
	//
	// All are treated the same: first we extract the value to the
	// widest representation (int64, uint64, float64, complex128,
	// or string), then we convert it to the desired type.

	switch ut_src := ut_src.(type) {
	case *types.Pointer:
		switch ut_dst := ut_dst.(type) {
		case *types.Basic:
			// *value to unsafe.Pointer?
			if ut_dst.Kind() == types.UnsafePointer {
				return unsafe.Pointer(x.(*value))
			}
		}

	case *types.Slice:
		// []byte or []rune -> string
		// TODO(adonovan): fix: type B byte; conv([]B -> string).
		switch ut_src.Elem().(*types.Basic).Kind() {
		case types.Byte:
			x := x.([]value)
			b := make([]byte, 0, len(x))
			for i := range x {
				b = append(b, x[i].(byte))
			}
			return string(b)

		case types.Rune:
			x := x.([]value)
			r := make([]rune, 0, len(x))
			for i := range x {
				r = append(r, x[i].(rune))
			}
			return string(r)
		}

	case *types.Basic:
		x = widen(x)

		// integer -> string?
		// TODO(adonovan): fix: test integer -> named alias of string.
		if ut_src.Info()&types.IsInteger != 0 {
			if ut_dst, ok := ut_dst.(*types.Basic); ok && ut_dst.Kind() == types.String {
				return string(asInt(x))
			}
		}

		// string -> []rune, []byte or string?
		if s, ok := x.(string); ok {
			switch ut_dst := ut_dst.(type) {
			case *types.Slice:
				var res []value
				// TODO(adonovan): fix: test named alias of rune, byte.
				switch ut_dst.Elem().(*types.Basic).Kind() {
				case types.Rune:
					for _, r := range []rune(s) {
						res = append(res, r)
					}
					return res
				case types.Byte:
					for _, b := range []byte(s) {
						res = append(res, b)
					}
					return res
				}
			case *types.Basic:
				if ut_dst.Kind() == types.String {
					return x.(string)
				}
			}
			break // fail: no other conversions for string
		}

		// unsafe.Pointer -> *value
		if ut_src.Kind() == types.UnsafePointer {
			// TODO(adonovan): this is wrong and cannot
			// really be fixed with the current design.
			//
			// return (*value)(x.(unsafe.Pointer))
			// creates a new pointer of a different
			// type but the underlying interface value
			// knows its "true" type and so cannot be
			// meaningfully used through the new pointer.
			//
			// To make this work, the interpreter needs to
			// simulate the memory layout of a real
			// compiled implementation.
			//
			// To at least preserve type-safety, we'll
			// just return the zero value of the
			// destination type.
			return zero(t_dst)
		}

		// Conversions between complex numeric types?
		if ut_src.Info()&types.IsComplex != 0 {
			switch ut_dst.(*types.Basic).Kind() {
			case types.Complex64:
				return complex64(x.(complex128))
			case types.Complex128:
				return x.(complex128)
			}
			break // fail: no other conversions for complex
		}

		// Conversions between non-complex numeric types?
		if ut_src.Info()&types.IsNumeric != 0 {
			kind := ut_dst.(*types.Basic).Kind()
			switch x := x.(type) {
			case int64: // signed integer -> numeric?
				switch kind {
				case types.Int:
					return int(x)
				case types.Int8:
					return int8(x)
				case types.Int16:
					return int16(x)
				case types.Int32:
					return int32(x)
				case types.Int64:
					return int64(x)
				case types.Uint:
					return uint(x)
				case types.Uint8:
					return uint8(x)
				case types.Uint16:
					return uint16(x)
				case types.Uint32:
					return uint32(x)
				case types.Uint64:
					return uint64(x)
				case types.Uintptr:
					return uintptr(x)
				case types.Float32:
					return float32(x)
				case types.Float64:
					return float64(x)
				}

			case uint64: // unsigned integer -> numeric?
				switch kind {
				case types.Int:
					return int(x)
				case types.Int8:
					return int8(x)
				case types.Int16:
					return int16(x)
				case types.Int32:
					return int32(x)
				case types.Int64:
					return int64(x)
				case types.Uint:
					return uint(x)
				case types.Uint8:
					return uint8(x)
				case types.Uint16:
					return uint16(x)
				case types.Uint32:
					return uint32(x)
				case types.Uint64:
					return uint64(x)
				case types.Uintptr:
					return uintptr(x)
				case types.Float32:
					return float32(x)
				case types.Float64:
					return float64(x)
				}

			case float64: // floating point -> numeric?
				switch kind {
				case types.Int:
					return int(x)
				case types.Int8:
					return int8(x)
				case types.Int16:
					return int16(x)
				case types.Int32:
					return int32(x)
				case types.Int64:
					return int64(x)
				case types.Uint:
					return uint(x)
				case types.Uint8:
					return uint8(x)
				case types.Uint16:
					return uint16(x)
				case types.Uint32:
					return uint32(x)
				case types.Uint64:
					return uint64(x)
				case types.Uintptr:
					return uintptr(x)
				case types.Float32:
					return float32(x)
				case types.Float64:
					return float64(x)
				}
			}
		}
	}

	panic(fmt.Sprintf("unsupported conversion: %s  -> %s, dynamic type %T", t_src, t_dst, x))
}

// checkInterface checks that the method set of x implements the
// interface itype.
// On success it returns "", on failure, an error message.
//
func checkInterface(i *interpreter, itype *types.Interface, x iface) string {
	if meth, _ := types.MissingMethod(x.t, itype, true); meth != nil {
		return fmt.Sprintf("interface conversion: %v is not %v: missing method %s",
			x.t, itype, meth.Name())
	}
	return "" // ok
}
