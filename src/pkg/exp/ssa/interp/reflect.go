package interp

// Emulated "reflect" package.
//
// We completely replace the built-in "reflect" package.
// The only thing clients can depend upon are that reflect.Type is an
// interface and reflect.Value is an (opaque) struct.

import (
	"exp/ssa"
	"fmt"
	"go/types"
	"reflect"
	"unsafe"
)

// A bogus "reflect" type-checker package.  Shared across interpreters.
var reflectTypesPackage = &types.Package{
	Name:     "reflect",
	Path:     "reflect",
	Complete: true,
}

// rtype is the concrete type the interpreter uses to implement the
// reflect.Type interface.  Since its type is opaque to the target
// language, we use a types.Basic.
//
// type rtype <opaque>
var rtypeType = makeNamedType("rtype", &types.Basic{Name: "rtype"})

func makeNamedType(name string, underlying types.Type) *types.NamedType {
	nt := &types.NamedType{Underlying: underlying}
	nt.Obj = &types.TypeName{
		Name: name,
		Type: nt,
		Pkg:  reflectTypesPackage,
	}
	return nt
}

func makeReflectValue(t types.Type, v value) value {
	return structure{rtype{t}, v}
}

// Given a reflect.Value, returns its rtype.
func rV2T(v value) rtype {
	return v.(structure)[0].(rtype)
}

// Given a reflect.Value, returns the underlying interpreter value.
func rV2V(v value) value {
	return v.(structure)[1]
}

// makeReflectType boxes up an rtype in a reflect.Type interface.
func makeReflectType(rt rtype) value {
	return iface{rtypeType, rt}
}

func ext۰reflect۰Init(fn *ssa.Function, args []value) value {
	// Signature: func()
	return nil
}

func ext۰reflect۰rtype۰Bits(fn *ssa.Function, args []value) value {
	// Signature: func (t reflect.rtype) int
	rt := args[0].(rtype).t
	basic, ok := underlyingType(rt).(*types.Basic)
	if !ok {
		panic(fmt.Sprintf("reflect.Type.Bits(%T): non-basic type", rt))
	}
	switch basic.Kind {
	case types.Int8, types.Uint8:
		return 8
	case types.Int16, types.Uint16:
		return 16
	case types.Int, types.UntypedInt:
		// Assume sizeof(int) is same on host and target; ditto uint.
		return reflect.TypeOf(int(0)).Bits()
	case types.Uintptr:
		// Assume sizeof(uintptr) is same on host and target.
		return reflect.TypeOf(uintptr(0)).Bits()
	case types.Int32, types.Uint32:
		return 32
	case types.Int64, types.Uint64:
		return 64
	case types.Float32:
		return 32
	case types.Float64, types.UntypedFloat:
		return 64
	case types.Complex64:
		return 64
	case types.Complex128, types.UntypedComplex:
		return 128
	default:
		panic(fmt.Sprintf("reflect.Type.Bits(%s)", basic))
	}
	return nil
}

func ext۰reflect۰rtype۰Elem(fn *ssa.Function, args []value) value {
	// Signature: func (t reflect.rtype) reflect.Type
	var elem types.Type
	switch rt := underlyingType(args[0].(rtype).t).(type) {
	case *types.Array:
		elem = rt.Elt
	case *types.Chan:
		elem = rt.Elt
	case *types.Map:
		elem = rt.Elt
	case *types.Pointer:
		elem = rt.Base
	case *types.Slice:
		elem = rt.Elt
	default:
		panic(fmt.Sprintf("reflect.Type.Elem(%T)", rt))
	}
	return makeReflectType(rtype{elem})
}

func ext۰reflect۰rtype۰Kind(fn *ssa.Function, args []value) value {
	// Signature: func (t reflect.rtype) uint
	return uint(reflectKind(args[0].(rtype).t))
}

func ext۰reflect۰rtype۰String(fn *ssa.Function, args []value) value {
	// Signature: func (t reflect.rtype) string
	return args[0].(rtype).t.String()
}

func ext۰reflect۰TypeOf(fn *ssa.Function, args []value) value {
	// Signature: func (t reflect.rtype) string
	return makeReflectType(rtype{args[0].(iface).t})
}

func ext۰reflect۰ValueOf(fn *ssa.Function, args []value) value {
	// Signature: func (interface{}) reflect.Value
	itf := args[0].(iface)
	return makeReflectValue(itf.t, itf.v)
}

func reflectKind(t types.Type) reflect.Kind {
	switch t := t.(type) {
	case *types.NamedType:
		return reflectKind(t.Underlying)
	case *types.Basic:
		switch t.Kind {
		case types.Bool:
			return reflect.Bool
		case types.Int:
			return reflect.Int
		case types.Int8:
			return reflect.Int8
		case types.Int16:
			return reflect.Int16
		case types.Int32:
			return reflect.Int32
		case types.Int64:
			return reflect.Int64
		case types.Uint:
			return reflect.Uint
		case types.Uint8:
			return reflect.Uint8
		case types.Uint16:
			return reflect.Uint16
		case types.Uint32:
			return reflect.Uint32
		case types.Uint64:
			return reflect.Uint64
		case types.Uintptr:
			return reflect.Uintptr
		case types.Float32:
			return reflect.Float32
		case types.Float64:
			return reflect.Float64
		case types.Complex64:
			return reflect.Complex64
		case types.Complex128:
			return reflect.Complex128
		case types.String:
			return reflect.String
		case types.UnsafePointer:
			return reflect.UnsafePointer
		}
	case *types.Array:
		return reflect.Array
	case *types.Chan:
		return reflect.Chan
	case *types.Signature:
		return reflect.Func
	case *types.Interface:
		return reflect.Interface
	case *types.Map:
		return reflect.Map
	case *types.Pointer:
		return reflect.Ptr
	case *types.Slice:
		return reflect.Slice
	case *types.Struct:
		return reflect.Struct
	}
	panic(fmt.Sprint("unexpected type: ", t))
}

func ext۰reflect۰Value۰Kind(fn *ssa.Function, args []value) value {
	// Signature: func (reflect.Value) uint
	return uint(reflectKind(rV2T(args[0]).t))
}

func ext۰reflect۰Value۰String(fn *ssa.Function, args []value) value {
	// Signature: func (reflect.Value) string
	return toString(rV2V(args[0]))
}

func ext۰reflect۰Value۰Type(fn *ssa.Function, args []value) value {
	// Signature: func (reflect.Value) reflect.Type
	return makeReflectType(rV2T(args[0]))
}

func ext۰reflect۰Value۰Len(fn *ssa.Function, args []value) value {
	// Signature: func (reflect.Value) int
	switch v := rV2V(args[0]).(type) {
	case string:
		return len(v)
	case array:
		return len(v)
	case chan value:
		return cap(v)
	case []value:
		return len(v)
	case *hashmap:
		return v.len()
	case map[value]value:
		return len(v)
	default:
		panic(fmt.Sprintf("reflect.(Value).Len(%V)", v))
	}
	return nil // unreachable
}

func ext۰reflect۰Value۰NumField(fn *ssa.Function, args []value) value {
	// Signature: func (reflect.Value) int
	return len(rV2V(args[0]).(structure))
}

func ext۰reflect۰Value۰Pointer(fn *ssa.Function, args []value) value {
	// Signature: func (v reflect.Value) uintptr
	switch v := rV2V(args[0]).(type) {
	case *value:
		return uintptr(unsafe.Pointer(v))
	case chan value:
		return reflect.ValueOf(v).Pointer()
	case []value:
		return reflect.ValueOf(v).Pointer()
	case *hashmap:
		return reflect.ValueOf(v.table).Pointer()
	case map[value]value:
		return reflect.ValueOf(v).Pointer()
	case *ssa.Function:
		return uintptr(unsafe.Pointer(v))
	default:
		panic(fmt.Sprintf("reflect.(Value).Pointer(%T)", v))
	}
	return nil // unreachable
}

func ext۰reflect۰Value۰Index(fn *ssa.Function, args []value) value {
	// Signature: func (v reflect.Value, i int) Value
	i := args[1].(int)
	t := underlyingType(rV2T(args[0]).t)
	switch v := rV2V(args[0]).(type) {
	case array:
		return makeReflectValue(t.(*types.Array).Elt, v[i])
	case []value:
		return makeReflectValue(t.(*types.Slice).Elt, v[i])
	default:
		panic(fmt.Sprintf("reflect.(Value).Index(%T)", v))
	}
	return nil // unreachable
}

func ext۰reflect۰Value۰CanAddr(fn *ssa.Function, args []value) value {
	// Signature: func (v reflect.Value) bool
	// Always false for our representation.
	return false
}

func ext۰reflect۰Value۰CanInterface(fn *ssa.Function, args []value) value {
	// Signature: func (v reflect.Value) bool
	// Always true for our representation.
	return true
}

func ext۰reflect۰Value۰Elem(fn *ssa.Function, args []value) value {
	// Signature: func (v reflect.Value) reflect.Value
	switch x := rV2V(args[0]).(type) {
	case iface:
		return makeReflectValue(x.t, x.v)
	case *value:
		return makeReflectValue(underlyingType(rV2T(args[0]).t).(*types.Pointer).Base, *x)
	default:
		panic(fmt.Sprintf("reflect.(Value).Elem(%T)", x))
	}
	return nil // unreachable
}

func ext۰reflect۰Value۰Field(fn *ssa.Function, args []value) value {
	// Signature: func (v reflect.Value, i int) reflect.Value
	v := args[0]
	i := args[1].(int)
	return makeReflectValue(underlyingType(rV2T(v).t).(*types.Struct).Fields[i].Type, rV2V(v).(structure)[i])
}

func ext۰reflect۰Value۰Interface(fn *ssa.Function, args []value) value {
	// Signature: func (v reflect.Value) interface{}
	return ext۰reflect۰valueInterface(fn, args)
}

func ext۰reflect۰Value۰Int(fn *ssa.Function, args []value) value {
	// Signature: func (reflect.Value) int64
	switch x := rV2V(args[0]).(type) {
	case int:
		return int64(x)
	case int8:
		return int64(x)
	case int16:
		return int64(x)
	case int32:
		return int64(x)
	case int64:
		return x
	default:
		panic(fmt.Sprintf("reflect.(Value).Int(%T)", x))
	}
	return nil // unreachable
}

func ext۰reflect۰Value۰IsNil(fn *ssa.Function, args []value) value {
	// Signature: func (reflect.Value) bool
	switch x := rV2V(args[0]).(type) {
	case *value:
		return x == nil
	case chan value:
		return x == nil
	case map[value]value:
		return x == nil
	case *hashmap:
		return x == nil
	case iface:
		return x.t == nil
	case []value:
		return x == nil
	case *ssa.Function:
		return x == nil
	case *ssa.Builtin:
		return x == nil
	case *closure:
		return x == nil
	default:
		panic(fmt.Sprintf("reflect.(Value).IsNil(%T)", x))
	}
	return nil // unreachable
}

func ext۰reflect۰Value۰IsValid(fn *ssa.Function, args []value) value {
	// Signature: func (reflect.Value) bool
	return rV2V(args[0]) != nil
}

func ext۰reflect۰valueInterface(fn *ssa.Function, args []value) value {
	// Signature: func (v reflect.Value, safe bool) interface{}
	v := args[0].(structure)
	return iface{rV2T(v).t, rV2V(v)}
}

// newMethod creates a new method of the specified name, package and receiver type.
func newMethod(pkg *ssa.Package, recvType types.Type, name string) *ssa.Function {
	fn := &ssa.Function{
		Name_: name,
		Pkg:   pkg,
		Prog:  pkg.Prog,
	}
	// TODO(adonovan): fix: hack: currently the only part of Signature
	// that is needed is the "pointerness" of Recv.Type, and for
	// now, we'll set it to always be false since we're only
	// concerned with rtype.  Encapsulate this better.
	fn.Signature = &types.Signature{Recv: &types.Var{
		Name: "recv",
		Type: recvType,
	}}
	return fn
}

func initReflect(i *interpreter) {
	i.reflectPackage = &ssa.Package{
		Prog:       i.prog,
		Types:      reflectTypesPackage,
		ImportPath: "reflect",
		Members:    make(map[string]ssa.Member),
	}

	i.rtypeMethods = ssa.MethodSet{
		ssa.Id{nil, "Bits"}:   newMethod(i.reflectPackage, rtypeType, "Bits"),
		ssa.Id{nil, "Elem"}:   newMethod(i.reflectPackage, rtypeType, "Elem"),
		ssa.Id{nil, "Kind"}:   newMethod(i.reflectPackage, rtypeType, "Kind"),
		ssa.Id{nil, "String"}: newMethod(i.reflectPackage, rtypeType, "String"),
	}
}
