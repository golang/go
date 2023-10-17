// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package rttype allows the compiler to share type information with
// the runtime. The shared type information is stored in
// internal/abi. This package translates those types from the host
// machine on which the compiler runs to the target machine on which
// the compiled program will run. In particular, this package handles
// layout differences between e.g. a 64 bit compiler and 32 bit
// target.
package rttype

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/objw"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"fmt"
	"internal/abi"
	"reflect"
)

type RuntimeType struct {
	// A *types.Type representing a type used at runtime.
	t *types.Type
	// components maps from component names to their location in the type.
	components map[string]location
}

type location struct {
	offset int64
	kind   types.Kind // Just used for bug detection
}

// Types shared with the runtime via internal/abi.
// TODO: add more
var Type *RuntimeType

func Init() {
	// Note: this has to be called explicitly instead of being
	// an init function so it runs after the types package has
	// been properly initialized.
	Type = fromReflect(reflect.TypeOf(abi.Type{}))

	// Make sure abi functions are correct. These functions are used
	// by the linker which doesn't have the ability to do type layout,
	// so we check the functions it uses here.
	ptrSize := types.PtrSize
	if got, want := int64(abi.CommonSize(ptrSize)), Type.Size(); got != want {
		base.Fatalf("abi.CommonSize() == %d, want %d", got, want)
	}
	if got, want := int64(abi.TFlagOff(ptrSize)), Type.Offset("TFlag"); got != want {
		base.Fatalf("abi.TFlagOff() == %d, want %d", got, want)
	}
}

// fromReflect translates from a host type to the equivalent
// target type.
func fromReflect(rt reflect.Type) *RuntimeType {
	t := reflectToType(rt)
	types.CalcSize(t)
	return &RuntimeType{t: t, components: unpack(t)}
}

// reflectToType converts from a reflect.Type (which is a compiler
// host type) to a *types.Type, which is a target type.  The result
// must be CalcSize'd before using.
func reflectToType(rt reflect.Type) *types.Type {
	switch rt.Kind() {
	case reflect.Bool:
		return types.Types[types.TBOOL]
	case reflect.Int:
		return types.Types[types.TINT]
	case reflect.Int32:
		return types.Types[types.TINT32]
	case reflect.Uint8:
		return types.Types[types.TUINT8]
	case reflect.Uint16:
		return types.Types[types.TUINT16]
	case reflect.Uint32:
		return types.Types[types.TUINT32]
	case reflect.Uintptr:
		return types.Types[types.TUINTPTR]
	case reflect.Ptr, reflect.Func, reflect.UnsafePointer:
		// TODO: there's no mechanism to distinguish different pointer types,
		// so we treat them all as unsafe.Pointer.
		return types.Types[types.TUNSAFEPTR]
	case reflect.Array:
		return types.NewArray(reflectToType(rt.Elem()), int64(rt.Len()))
	case reflect.Struct:
		fields := make([]*types.Field, rt.NumField())
		for i := 0; i < rt.NumField(); i++ {
			f := rt.Field(i)
			ft := reflectToType(f.Type)
			fields[i] = &types.Field{Sym: &types.Sym{Name: f.Name}, Type: ft}
		}
		return types.NewStruct(fields)
	default:
		base.Fatalf("unhandled kind %s", rt.Kind())
		return nil
	}
}

// Unpack generates a set of components of a *types.Type.
// The type must have already been CalcSize'd.
func unpack(t *types.Type) map[string]location {
	components := map[string]location{}
	switch t.Kind() {
	default:
		components[""] = location{0, t.Kind()}
	case types.TARRAY:
		// TODO: not used yet
		elemSize := t.Elem().Size()
		for name, loc := range unpack(t.Elem()) {
			for i := int64(0); i < t.NumElem(); i++ {
				components[fmt.Sprintf("[%d]%s", i, name)] = location{i*elemSize + loc.offset, loc.kind}
			}
		}
	case types.TSTRUCT:
		for _, f := range t.Fields() {
			for name, loc := range unpack(f.Type) {
				n := f.Sym.Name
				if name != "" {
					n += "." + name
				}
				components[n] = location{f.Offset + loc.offset, loc.kind}
			}
		}
	}
	return components
}

func (r *RuntimeType) Size() int64 {
	return r.t.Size()
}

func (r *RuntimeType) Alignment() int64 {
	return r.t.Alignment()
}

func (r *RuntimeType) Offset(name string) int64 {
	return r.components[name].offset
}

// WritePtr writes a pointer "target" to the component named "name" in the
// static object "lsym".
func (r *RuntimeType) WritePtr(lsym *obj.LSym, name string, target *obj.LSym) {
	loc := r.components[name]
	if loc.kind != types.TUNSAFEPTR {
		base.Fatalf("can't write ptr to field %s, it has kind %s", name, loc.kind)
	}
	if target == nil {
		objw.Uintptr(lsym, int(loc.offset), 0)
	} else {
		objw.SymPtr(lsym, int(loc.offset), target, 0)
	}
}
func (r *RuntimeType) WriteUintptr(lsym *obj.LSym, name string, val uint64) {
	loc := r.components[name]
	if loc.kind != types.TUINTPTR {
		base.Fatalf("can't write uintptr to field %s, it has kind %s", name, loc.kind)
	}
	objw.Uintptr(lsym, int(loc.offset), val)
}
func (r *RuntimeType) WriteUint32(lsym *obj.LSym, name string, val uint32) {
	loc := r.components[name]
	if loc.kind != types.TUINT32 {
		base.Fatalf("can't write uint32 to field %s, it has kind %s", name, loc.kind)
	}
	objw.Uint32(lsym, int(loc.offset), val)
}
func (r *RuntimeType) WriteUint8(lsym *obj.LSym, name string, val uint8) {
	loc := r.components[name]
	if loc.kind != types.TUINT8 {
		base.Fatalf("can't write uint8 to field %s, it has kind %s", name, loc.kind)
	}
	objw.Uint8(lsym, int(loc.offset), val)
}
func (r *RuntimeType) WriteSymPtrOff(lsym *obj.LSym, name string, target *obj.LSym, weak bool) {
	loc := r.components[name]
	if loc.kind != types.TINT32 {
		base.Fatalf("can't write SymPtr to field %s, it has kind %s", name, loc.kind)
	}
	if target == nil {
		objw.Uint32(lsym, int(loc.offset), 0)
	} else if weak {
		objw.SymPtrWeakOff(lsym, int(loc.offset), target)
	} else {
		objw.SymPtrOff(lsym, int(loc.offset), target)
	}
}
