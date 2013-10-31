// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package interp

// Emulated "reflect" package.
//
// We completely replace the built-in "reflect" package.
// The only thing clients can depend upon are that reflect.Type is an
// interface and reflect.Value is an (opaque) struct.

import (
	"fmt"
	"go/token"
	"reflect"
	"unsafe"

	"code.google.com/p/go.tools/go/types"
	"code.google.com/p/go.tools/ssa"
)

type opaqueType struct {
	types.Type
	name string
}

func (t *opaqueType) String() string { return t.name }

// A bogus "reflect" type-checker package.  Shared across interpreters.
var reflectTypesPackage = types.NewPackage("reflect", "reflect", nil)

// rtype is the concrete type the interpreter uses to implement the
// reflect.Type interface.  Since its type is opaque to the target
// language, we use a types.Basic.
//
// type rtype <opaque>
var rtypeType = makeNamedType("rtype", &opaqueType{nil, "rtype"})

// error is an (interpreted) named type whose underlying type is string.
// The interpreter uses it for all implementations of the built-in error
// interface that it creates.
// We put it in the "reflect" package for expedience.
//
// type error string
var errorType = makeNamedType("error", &opaqueType{nil, "error"})

func makeNamedType(name string, underlying types.Type) *types.Named {
	obj := types.NewTypeName(token.NoPos, reflectTypesPackage, name, nil)
	return types.NewNamed(obj, underlying, nil)
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
	basic, ok := rt.Underlying().(*types.Basic)
	if !ok {
		panic(fmt.Sprintf("reflect.Type.Bits(%T): non-basic type", rt))
	}
	switch basic.Kind() {
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
	return makeReflectType(rtype{args[0].(rtype).t.Underlying().(interface {
		Elem() types.Type
	}).Elem()})
}

func ext۰reflect۰rtype۰Field(fn *ssa.Function, args []value) value {
	// Signature: func (t reflect.rtype, i int) reflect.StructField
	st := args[0].(rtype).t.Underlying().(*types.Struct)
	i := args[1].(int)
	f := st.Field(i)
	return structure{
		f.Name(),
		f.Pkg().Path(),
		makeReflectType(rtype{f.Type()}),
		st.Tag(i),
		0,         // TODO(adonovan): offset
		[]value{}, // TODO(adonovan): indices
		f.Anonymous(),
	}
}

func ext۰reflect۰rtype۰Kind(fn *ssa.Function, args []value) value {
	// Signature: func (t reflect.rtype) uint
	return uint(reflectKind(args[0].(rtype).t))
}

func ext۰reflect۰rtype۰NumField(fn *ssa.Function, args []value) value {
	// Signature: func (t reflect.rtype) int
	return args[0].(rtype).t.Underlying().(*types.Struct).NumFields()
}

func ext۰reflect۰rtype۰NumMethod(fn *ssa.Function, args []value) value {
	// Signature: func (t reflect.rtype) int
	return args[0].(rtype).t.MethodSet().Len()
}

func ext۰reflect۰rtype۰NumOut(fn *ssa.Function, args []value) value {
	// Signature: func (t reflect.rtype) int
	return args[0].(rtype).t.(*types.Signature).Results().Len()
}

func ext۰reflect۰rtype۰Out(fn *ssa.Function, args []value) value {
	// Signature: func (t reflect.rtype, i int) int
	i := args[1].(int)
	return makeReflectType(rtype{args[0].(rtype).t.(*types.Signature).Results().At(i).Type()})
}

func ext۰reflect۰rtype۰Size(fn *ssa.Function, args []value) value {
	// Signature: func (t reflect.rtype) uintptr
	// (Assumes no custom Sizeof used during SSA construction.)
	return uintptr(stdSizes.Sizeof(args[0].(rtype).t))
}

func ext۰reflect۰rtype۰String(fn *ssa.Function, args []value) value {
	// Signature: func (t reflect.rtype) string
	return args[0].(rtype).t.String()
}

func ext۰reflect۰New(fn *ssa.Function, args []value) value {
	// Signature: func (t reflect.Type) reflect.Value
	t := args[0].(iface).v.(rtype).t
	alloc := zero(t)
	return makeReflectValue(types.NewPointer(t), &alloc)
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
	case *types.Named:
		return reflectKind(t.Underlying())
	case *types.Basic:
		switch t.Kind() {
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

func ext۰reflect۰Value۰Uint(fn *ssa.Function, args []value) value {
	// Signature: func (reflect.Value) uint64
	switch v := rV2V(args[0]).(type) {
	case uint:
		return uint64(v)
	case uint8:
		return uint64(v)
	case uint16:
		return uint64(v)
	case uint32:
		return uint64(v)
	case uint64:
		return uint64(v)
	case uintptr:
		return uint64(v)
	}
	panic("reflect.Value.Uint")
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
		panic(fmt.Sprintf("reflect.(Value).Len(%v)", v))
	}
	return nil // unreachable
}

func ext۰reflect۰Value۰MapIndex(fn *ssa.Function, args []value) value {
	// Signature: func (reflect.Value) Value
	tValue := rV2T(args[0]).t.Underlying().(*types.Map).Key()
	k := rV2V(args[1])
	switch m := rV2V(args[0]).(type) {
	case map[value]value:
		if v, ok := m[k]; ok {
			return makeReflectValue(tValue, v)
		}

	case *hashmap:
		if v := m.lookup(k.(hashable)); v != nil {
			return makeReflectValue(tValue, v)
		}

	default:
		panic(fmt.Sprintf("(reflect.Value).MapIndex(%T, %T)", m, k))
	}
	return makeReflectValue(nil, nil)
}

func ext۰reflect۰Value۰MapKeys(fn *ssa.Function, args []value) value {
	// Signature: func (reflect.Value) []Value
	var keys []value
	tKey := rV2T(args[0]).t.Underlying().(*types.Map).Key()
	switch v := rV2V(args[0]).(type) {
	case map[value]value:
		for k := range v {
			keys = append(keys, makeReflectValue(tKey, k))
		}

	case *hashmap:
		for _, e := range v.table {
			for ; e != nil; e = e.next {
				keys = append(keys, makeReflectValue(tKey, e.key))
			}
		}

	default:
		panic(fmt.Sprintf("(reflect.Value).MapKeys(%T)", v))
	}
	return keys
}

func ext۰reflect۰Value۰NumField(fn *ssa.Function, args []value) value {
	// Signature: func (reflect.Value) int
	return len(rV2V(args[0]).(structure))
}

func ext۰reflect۰Value۰NumMethod(fn *ssa.Function, args []value) value {
	// Signature: func (reflect.Value) int
	return rV2T(args[0]).t.MethodSet().Len()
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
	t := rV2T(args[0]).t.Underlying()
	switch v := rV2V(args[0]).(type) {
	case array:
		return makeReflectValue(t.(*types.Array).Elem(), v[i])
	case []value:
		return makeReflectValue(t.(*types.Slice).Elem(), v[i])
	default:
		panic(fmt.Sprintf("reflect.(Value).Index(%T)", v))
	}
	return nil // unreachable
}

func ext۰reflect۰Value۰Bool(fn *ssa.Function, args []value) value {
	// Signature: func (reflect.Value) bool
	return rV2V(args[0]).(bool)
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
		return makeReflectValue(rV2T(args[0]).t.Underlying().(*types.Pointer).Elem(), *x)
	default:
		panic(fmt.Sprintf("reflect.(Value).Elem(%T)", x))
	}
	return nil // unreachable
}

func ext۰reflect۰Value۰Field(fn *ssa.Function, args []value) value {
	// Signature: func (v reflect.Value, i int) reflect.Value
	v := args[0]
	i := args[1].(int)
	return makeReflectValue(rV2T(v).t.Underlying().(*types.Struct).Field(i).Type(), rV2V(v).(structure)[i])
}

func ext۰reflect۰Value۰Float(fn *ssa.Function, args []value) value {
	// Signature: func (reflect.Value) float64
	switch v := rV2V(args[0]).(type) {
	case float32:
		return float64(v)
	case float64:
		return float64(v)
	}
	panic("reflect.Value.Float")
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

func ext۰reflect۰Value۰Set(fn *ssa.Function, args []value) value {
	// TODO(adonovan): implement.
	return nil
}

func ext۰reflect۰valueInterface(fn *ssa.Function, args []value) value {
	// Signature: func (v reflect.Value, safe bool) interface{}
	v := args[0].(structure)
	return iface{rV2T(v).t, rV2V(v)}
}

func ext۰reflect۰error۰Error(fn *ssa.Function, args []value) value {
	return args[0]
}

// newMethod creates a new method of the specified name, package and receiver type.
func newMethod(pkg *ssa.Package, recvType types.Type, name string) *ssa.Function {
	// TODO(adonovan): fix: hack: currently the only part of Signature
	// that is needed is the "pointerness" of Recv.Type, and for
	// now, we'll set it to always be false since we're only
	// concerned with rtype.  Encapsulate this better.
	sig := types.NewSignature(nil, types.NewVar(token.NoPos, nil, "recv", recvType), nil, nil, false)
	fn := ssa.NewFunction(name, sig, "fake reflect method")
	fn.Pkg = pkg
	fn.Prog = pkg.Prog
	return fn
}

func initReflect(i *interpreter) {
	i.reflectPackage = &ssa.Package{
		Prog:    i.prog,
		Object:  reflectTypesPackage,
		Members: make(map[string]ssa.Member),
	}

	i.rtypeMethods = methodSet{
		"Bits":      newMethod(i.reflectPackage, rtypeType, "Bits"),
		"Elem":      newMethod(i.reflectPackage, rtypeType, "Elem"),
		"Field":     newMethod(i.reflectPackage, rtypeType, "Field"),
		"Kind":      newMethod(i.reflectPackage, rtypeType, "Kind"),
		"NumField":  newMethod(i.reflectPackage, rtypeType, "NumField"),
		"NumMethod": newMethod(i.reflectPackage, rtypeType, "NumMethod"),
		"NumOut":    newMethod(i.reflectPackage, rtypeType, "NumOut"),
		"Out":       newMethod(i.reflectPackage, rtypeType, "Out"),
		"Size":      newMethod(i.reflectPackage, rtypeType, "Size"),
		"String":    newMethod(i.reflectPackage, rtypeType, "String"),
	}
	i.errorMethods = methodSet{
		"Error": newMethod(i.reflectPackage, errorType, "Error"),
	}
}
