package interp

import (
	"exp/ssa"
	"fmt"
	"go/token"
	"go/types"
	"os"
	"runtime"
	"strings"
	"unsafe"
)

// If the target program panics, the interpreter panics with this type.
type targetPanic struct {
	v value
}

// literalValue returns the value of the literal with the
// dynamic type tag appropriate for l.Type().
func literalValue(l *ssa.Literal) value {
	if l.IsNil() {
		return zero(l.Type()) // typed nil
	}

	// By destination type:
	switch t := underlyingType(l.Type()).(type) {
	case *types.Basic:
		switch t.Kind {
		case types.Bool, types.UntypedBool:
			return l.Value.(bool)
		case types.Int, types.UntypedInt:
			// Assume sizeof(int) is same on host and target.
			return int(l.Int64())
		case types.Int8:
			return int8(l.Int64())
		case types.Int16:
			return int16(l.Int64())
		case types.Int32, types.UntypedRune:
			return int32(l.Int64())
		case types.Int64:
			return l.Int64()
		case types.Uint:
			// Assume sizeof(uint) is same on host and target.
			return uint(l.Uint64())
		case types.Uint8:
			return uint8(l.Uint64())
		case types.Uint16:
			return uint16(l.Uint64())
		case types.Uint32:
			return uint32(l.Uint64())
		case types.Uint64:
			return l.Uint64()
		case types.Uintptr:
			// Assume sizeof(uintptr) is same on host and target.
			return uintptr(l.Uint64())
		case types.Float32:
			return float32(l.Float64())
		case types.Float64, types.UntypedFloat:
			return l.Float64()
		case types.Complex64:
			return complex64(l.Complex128())
		case types.Complex128, types.UntypedComplex:
			return l.Complex128()
		case types.String, types.UntypedString:
			if v, ok := l.Value.(string); ok {
				return v
			}
			return string(rune(l.Int64()))
		case types.UnsafePointer:
			panic("unsafe.Pointer literal") // not possible
		case types.UntypedNil:
			// nil was handled above.
		}

	case *types.Slice:
		switch et := underlyingType(t.Elt).(type) {
		case *types.Basic:
			switch et.Kind {
			case types.Byte: // string -> []byte
				var v []value
				for _, b := range []byte(l.Value.(string)) {
					v = append(v, b)
				}
				return v
			case types.Rune: // string -> []rune
				var v []value
				for _, r := range []rune(l.Value.(string)) {
					v = append(v, r)
				}
				return v
			}
		}
	}

	panic(fmt.Sprintf("literalValue: Value.(type)=%T Type()=%s", l.Value, l.Type()))
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
		if t.Kind == types.UntypedNil {
			panic("untyped nil has no zero value")
		}
		if t.Info&types.IsUntyped != 0 {
			t = ssa.DefaultType(t).(*types.Basic)
		}
		switch t.Kind {
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
		a := make(array, t.Len)
		for i := range a {
			a[i] = zero(t.Elt)
		}
		return a
	case *types.NamedType:
		return zero(t.Underlying)
	case *types.Interface:
		return iface{} // nil type, methodset and value
	case *types.Slice:
		return []value(nil)
	case *types.Struct:
		s := make(structure, len(t.Fields))
		for i := range s {
			s[i] = zero(t.Fields[i].Type)
		}
		return s
	case *types.Chan:
		return chan value(nil)
	case *types.Map:
		if usesBuiltinMap(t.Key) {
			return map[value]value(nil)
		}
		return (*hashmap)(nil)

	case *types.Signature:
		return (*ssa.Function)(nil)
	}
	panic(fmt.Sprint("zero: unexpected ", t))
}

// slice returns x[lo:hi].  Either or both of lo and hi may be nil.
func slice(x, lo, hi value) value {
	l := 0
	if lo != nil {
		l = asInt(lo)
	}
	switch x := x.(type) {
	case string:
		if hi != nil {
			return x[l:asInt(hi)]
		}
		return x[l:]
	case []value:
		if hi != nil {
			return x[l:asInt(hi)]
		}
		return x[l:]
	case *value: // *array
		a := (*x).(array)
		if hi != nil {
			return []value(a)[l:asInt(hi)]
		}
		return []value(a)[l:]
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
		if ok {
			v = copyVal(v)
		} else {
			v = zero(underlyingType(instr.X.Type()).(*types.Map).Elt)
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
func binop(op token.Token, x, y value) value {
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
		return equals(x, y)

	case token.NEQ:
		return !equals(x, y)

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

func unop(instr *ssa.UnOp, x value) value {
	switch instr.Op {
	case token.ARROW: // receive
		v, ok := <-x.(chan value)
		if !ok {
			v = zero(underlyingType(instr.X.Type()).(*types.Chan).Elt)
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
		}
	case token.MUL:
		return copyVal(*x.(*value)) // load
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
	if idst, ok := underlyingType(instr.AssertedType).(*types.Interface); ok {
		v = itf
		err = checkInterface(i, idst, itf)

	} else if types.IsIdentical(itf.t, instr.AssertedType) {
		v = copyVal(itf.v) // extract value

	} else {
		err = fmt.Sprintf("type assert failed: expected %s, got %s", instr.AssertedType, itf.t)
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

	case "copy": // copy([]T, []T) int
		if _, ok := args[1].(string); ok {
			panic("copy([]byte, string) not yet implemented")
		}
		return copy(args[0].([]value), args[1].([]value))

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

	case "print", "println": // print(interface{}, ...interface{})
		ln := fn.Name() == "println"
		fmt.Print(toString(args[0]))
		if len(args) == 2 {
			for _, arg := range args[1].([]value) {
				if ln {
					fmt.Print(" ")
				}
				fmt.Print(toString(arg))
			}
		}
		if ln {
			fmt.Println()
		}
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
		panic(targetPanic{args[0]})

	case "recover":
		// recover() must be exactly one level beneath the
		// deferred function (two levels beneath the panicking
		// function) to have any effect.  Thus we ignore both
		// "defer recover()" and "defer f() -> g() ->
		// recover()".
		if caller.i.mode&DisableRecover == 0 &&
			caller != nil && caller.status == stRunning &&
			caller.caller != nil && caller.caller.status == stPanic {
			caller.caller.status = stComplete
			p := caller.caller.panic
			caller.caller.panic = nil
			switch p := p.(type) {
			case targetPanic:
				return p.v
			case runtime.Error:
				// TODO(adonovan): must box this up
				// inside instance of interface 'error'.
				return iface{types.Typ[types.String], p.Error()}
			case string:
				return iface{types.Typ[types.String], p}
			default:
				panic(fmt.Sprintf("unexpected panic type %T in target call to recover()", p))
			}
		}
		return iface{}
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
// the result.  Possible cases are described with the ssa.Conv
// operator.  Panics if the dynamic conversion fails.
//
func conv(t_dst, t_src types.Type, x value) value {
	ut_src := underlyingType(t_src)
	ut_dst := underlyingType(t_dst)

	// Same underlying types?
	// TODO(adonovan): consider a dedicated ssa.ChangeType instruction.
	if types.IsIdentical(ut_dst, ut_src) {
		return x
	}

	// Destination type is not an "untyped" type.
	if b, ok := ut_dst.(*types.Basic); ok && b.Info&types.IsUntyped != 0 {
		panic("conversion to 'untyped' type: " + b.String())
	}

	// Nor is it an interface type.
	if _, ok := ut_dst.(*types.Interface); ok {
		if _, ok := ut_src.(*types.Interface); ok {
			panic("oops: Conv should be ChangeInterface")
		} else {
			panic("oops: Conv should be MakeInterface")
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
	// widest representation (bool, int64, uint64, float64,
	// complex128, or string), then we convert it to the desired
	// type.

	switch ut_src := ut_src.(type) {
	case *types.Signature:
		// TODO(adonovan): fix: this is a hacky workaround for the
		// unsound conversion of Signature types from
		// func(T)() to func()(T), i.e. arg0 <-> receiver
		// conversion.  Talk to gri about correct approach.
		fmt.Fprintln(os.Stderr, "Warning: unsound Signature conversion")
		return x

	case *types.Pointer:
		// *value to unsafe.Pointer?
		if ut_dst, ok := ut_dst.(*types.Basic); ok {
			if ut_dst.Kind == types.UnsafePointer {
				return unsafe.Pointer(x.(*value))
			}
		}

	case *types.Slice:
		// []byte or []rune -> string
		// TODO(adonovan): fix: type B byte; conv([]B -> string).
		switch ut_src.Elt.(*types.Basic).Kind {
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

		// bool?
		if _, ok := x.(bool); ok {
			return x
		}

		// integer -> string?
		// TODO(adonovan): fix: test integer -> named alias of string.
		if ut_src.Info&types.IsInteger != 0 {
			if ut_dst, ok := ut_dst.(*types.Basic); ok && ut_dst.Kind == types.String {
				return string(asInt(x))
			}
		}

		// string -> []rune, []byte or string?
		if s, ok := x.(string); ok {
			switch ut_dst := ut_dst.(type) {
			case *types.Slice:
				var res []value
				// TODO(adonovan): fix: test named alias of rune, byte.
				switch ut_dst.Elt.(*types.Basic).Kind {
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
				if ut_dst.Kind == types.String {
					return x.(string)
				}
			}
			break // fail: no other conversions for string
		}

		// unsafe.Pointer -> *value
		if ut_src.Kind == types.UnsafePointer {
			// TODO(adonovan): this is wrong and cannot
			// really be fixed with the current design.
			//
			// It creates a new pointer of a different
			// type but the underlying interface value
			// knows its "true" type and so cannot be
			// meaningfully used through the new pointer.
			//
			// To make this work, the interpreter needs to
			// simulate the memory layout of a real
			// compiled implementation.
			return (*value)(x.(unsafe.Pointer))
		}

		// Conversions between complex numeric types?
		if ut_src.Info&types.IsComplex != 0 {
			switch ut_dst.(*types.Basic).Kind {
			case types.Complex64:
				return complex64(x.(complex128))
			case types.Complex128:
				return x.(complex128)
			}
			break // fail: no other conversions for complex
		}

		// Conversions between non-complex numeric types?
		if ut_src.Info&types.IsNumeric != 0 {
			kind := ut_dst.(*types.Basic).Kind
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
func checkInterface(i *interpreter, itype types.Type, x iface) string {
	mset := findMethodSet(i, x.t)
	for _, m := range underlyingType(itype).(*types.Interface).Methods {
		id := ssa.IdFromQualifiedName(m.QualifiedName)
		if mset[id] == nil {
			return fmt.Sprintf("interface conversion: %v is not %v: missing method %v", x.t, itype, id)
		}
	}
	return "" // ok
}

// underlyingType returns the underlying type of typ.
// Copied from go/types.underlying.
//
func underlyingType(typ types.Type) types.Type {
	if typ, ok := typ.(*types.NamedType); ok {
		return typ.Underlying
	}
	return typ
}

// indirectType(typ) assumes that typ is a pointer type,
// or named alias thereof, and returns its base type.
// Panic ensues if it is not a pointer.
// Copied from exp/ssa.indirectType.
//
func indirectType(ptr types.Type) types.Type {
	return underlyingType(ptr).(*types.Pointer).Base
}
