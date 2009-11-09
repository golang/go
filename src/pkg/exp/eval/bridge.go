// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eval

import (
	"log";
	"go/token";
	"reflect";
)

/*
 * Type bridging
 */

var (
	evalTypes	= make(map[reflect.Type]Type);
	nativeTypes	= make(map[Type]reflect.Type);
)

// TypeFromNative converts a regular Go type into a the corresponding
// interpreter Type.
func TypeFromNative(t reflect.Type) Type {
	if et, ok := evalTypes[t]; ok {
		return et
	}

	var nt *NamedType;
	if t.Name() != "" {
		name := t.PkgPath() + "·" + t.Name();
		nt = &NamedType{token.Position{}, name, nil, true, make(map[string]Method)};
		evalTypes[t] = nt;
	}

	var et Type;
	switch t := t.(type) {
	case *reflect.BoolType:
		et = BoolType
	case *reflect.Float32Type:
		et = Float32Type
	case *reflect.Float64Type:
		et = Float64Type
	case *reflect.FloatType:
		et = FloatType
	case *reflect.Int16Type:
		et = Int16Type
	case *reflect.Int32Type:
		et = Int32Type
	case *reflect.Int64Type:
		et = Int64Type
	case *reflect.Int8Type:
		et = Int8Type
	case *reflect.IntType:
		et = IntType
	case *reflect.StringType:
		et = StringType
	case *reflect.Uint16Type:
		et = Uint16Type
	case *reflect.Uint32Type:
		et = Uint32Type
	case *reflect.Uint64Type:
		et = Uint64Type
	case *reflect.Uint8Type:
		et = Uint8Type
	case *reflect.UintType:
		et = UintType
	case *reflect.UintptrType:
		et = UintptrType

	case *reflect.ArrayType:
		et = NewArrayType(int64(t.Len()), TypeFromNative(t.Elem()))
	case *reflect.ChanType:
		log.Crashf("%T not implemented", t)
	case *reflect.FuncType:
		nin := t.NumIn();
		// Variadic functions have DotDotDotType at the end
		varidic := false;
		if nin > 0 {
			if _, ok := t.In(nin-1).(*reflect.DotDotDotType); ok {
				varidic = true;
				nin--;
			}
		}
		in := make([]Type, nin);
		for i := range in {
			in[i] = TypeFromNative(t.In(i))
		}
		out := make([]Type, t.NumOut());
		for i := range out {
			out[i] = TypeFromNative(t.Out(i))
		}
		et = NewFuncType(in, varidic, out);
	case *reflect.InterfaceType:
		log.Crashf("%T not implemented", t)
	case *reflect.MapType:
		log.Crashf("%T not implemented", t)
	case *reflect.PtrType:
		et = NewPtrType(TypeFromNative(t.Elem()))
	case *reflect.SliceType:
		et = NewSliceType(TypeFromNative(t.Elem()))
	case *reflect.StructType:
		n := t.NumField();
		fields := make([]StructField, n);
		for i := 0; i < n; i++ {
			sf := t.Field(i);
			// TODO(austin) What to do about private fields?
			fields[i].Name = sf.Name;
			fields[i].Type = TypeFromNative(sf.Type);
			fields[i].Anonymous = sf.Anonymous;
		}
		et = NewStructType(fields);
	case *reflect.UnsafePointerType:
		log.Crashf("%T not implemented", t)
	default:
		log.Crashf("unexpected reflect.Type: %T", t)
	}

	if nt != nil {
		if _, ok := et.(*NamedType); !ok {
			nt.Complete(et);
			et = nt;
		}
	}

	nativeTypes[et] = t;
	evalTypes[t] = et;

	return et;
}

// TypeOfNative returns the interpreter Type of a regular Go value.
func TypeOfNative(v interface{}) Type	{ return TypeFromNative(reflect.Typeof(v)) }

/*
 * Function bridging
 */

type nativeFunc struct {
	fn	func(*Thread, []Value, []Value);
	in, out	int;
}

func (f *nativeFunc) NewFrame() *Frame {
	vars := make([]Value, f.in + f.out);
	return &Frame{nil, vars};
}

func (f *nativeFunc) Call(t *Thread) {
	f.fn(t, t.f.Vars[0 : f.in], t.f.Vars[f.in : f.in + f.out])
}

// FuncFromNative creates an interpreter function from a native
// function that takes its in and out arguments as slices of
// interpreter Value's.  While somewhat inconvenient, this avoids
// value marshalling.
func FuncFromNative(fn func(*Thread, []Value, []Value), t *FuncType) FuncValue {
	return &funcV{&nativeFunc{fn, len(t.In), len(t.Out)}}
}

// FuncFromNativeTyped is like FuncFromNative, but constructs the
// function type from a function pointer using reflection.  Typically,
// the type will be given as a nil pointer to a function with the
// desired signature.
func FuncFromNativeTyped(fn func(*Thread, []Value, []Value), t interface{}) (*FuncType, FuncValue) {
	ft := TypeOfNative(t).(*FuncType);
	return ft, FuncFromNative(fn, ft);
}
