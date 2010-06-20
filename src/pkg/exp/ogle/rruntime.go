// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ogle

import (
	"debug/proc"
	"exp/eval"
	"reflect"
)

// This file contains remote runtime definitions.  Using reflection,
// we convert all of these to interpreter types and layout their
// remote representations using the architecture rules.
//
// We could get most of these definitions from our own runtime
// package; however, some of them differ in convenient ways, some of
// them are not defined or exported by the runtime, and having our own
// definitions makes it easy to support multiple remote runtime
// versions.  This may turn out to be overkill.
//
// All of these structures are prefixed with rt1 to indicate the
// runtime version and to mark them as types used only as templates
// for remote types.

/*
 * Runtime data headers
 *
 * See $GOROOT/src/pkg/runtime/runtime.h
 */

type rt1String struct {
	str uintptr
	len int
}

type rt1Slice struct {
	array uintptr
	len   int
	cap   int
}

type rt1Eface struct {
	typ uintptr
	ptr uintptr
}

/*
 * Runtime type structures
 *
 * See $GOROOT/src/pkg/runtime/type.h and $GOROOT/src/pkg/runtime/type.go
 */

type rt1UncommonType struct {
	name    *string
	pkgPath *string
	//methods []method;
}

type rt1CommonType struct {
	size                   uintptr
	hash                   uint32
	alg, align, fieldAlign uint8
	string                 *string
	uncommonType           *rt1UncommonType
}

type rt1Type struct {
	// While Type is technically an Eface, treating the
	// discriminator as an opaque pointer and taking advantage of
	// the commonType prologue on all Type's makes type parsing
	// much simpler.
	typ uintptr
	ptr *rt1CommonType
}

type rt1StructField struct {
	name    *string
	pkgPath *string
	typ     *rt1Type
	tag     *string
	offset  uintptr
}

type rt1StructType struct {
	rt1CommonType
	fields []rt1StructField
}

type rt1PtrType struct {
	rt1CommonType
	elem *rt1Type
}

type rt1SliceType struct {
	rt1CommonType
	elem *rt1Type
}

type rt1ArrayType struct {
	rt1CommonType
	elem *rt1Type
	len  uintptr
}

/*
 * Runtime scheduler structures
 *
 * See $GOROOT/src/pkg/runtime/runtime.h
 */

// Fields beginning with _ are only for padding

type rt1Stktop struct {
	stackguard uintptr
	stackbase  *rt1Stktop
	gobuf      rt1Gobuf
	_args      uint32
	_fp        uintptr
}

type rt1Gobuf struct {
	sp uintptr
	pc uintptr
	g  *rt1G
	r0 uintptr
}

type rt1G struct {
	_stackguard uintptr
	stackbase   *rt1Stktop
	_defer      uintptr
	sched       rt1Gobuf
	_stack0     uintptr
	_entry      uintptr
	alllink     *rt1G
	_param      uintptr
	status      int16
	// Incomplete
}

var rt1GStatus = runtimeGStatus{
	Gidle:     0,
	Grunnable: 1,
	Grunning:  2,
	Gsyscall:  3,
	Gwaiting:  4,
	Gmoribund: 5,
	Gdead:     6,
}

// runtimeIndexes stores the indexes of fields in the runtime
// structures.  It is filled in using reflection, so the name of the
// fields must match the names of the remoteType's in runtimeValues
// exactly and the names of the index fields must be the capitalized
// version of the names of the fields in the runtime structures above.
type runtimeIndexes struct {
	String struct {
		Str, Len int
	}
	Slice struct {
		Array, Len, Cap int
	}
	Eface struct {
		Typ, Ptr int
	}

	UncommonType struct {
		Name, PkgPath int
	}
	CommonType struct {
		Size, Hash, Alg, Align, FieldAlign, String, UncommonType int
	}
	Type struct {
		Typ, Ptr int
	}
	StructField struct {
		Name, PkgPath, Typ, Tag, Offset int
	}
	StructType struct {
		Fields int
	}
	PtrType struct {
		Elem int
	}
	SliceType struct {
		Elem int
	}
	ArrayType struct {
		Elem, Len int
	}

	Stktop struct {
		Stackguard, Stackbase, Gobuf int
	}
	Gobuf struct {
		Sp, Pc, G int
	}
	G struct {
		Stackbase, Sched, Status, Alllink int
	}
}

// Values of G status codes
type runtimeGStatus struct {
	Gidle, Grunnable, Grunning, Gsyscall, Gwaiting, Gmoribund, Gdead int64
}

// runtimeValues stores the types and values that correspond to those
// in the remote runtime package.
type runtimeValues struct {
	// Runtime data headers
	String, Slice, Eface *remoteType
	// Runtime type structures
	Type, CommonType, UncommonType, StructField, StructType, PtrType,
	ArrayType, SliceType *remoteType
	// Runtime scheduler structures
	Stktop, Gobuf, G *remoteType
	// Addresses of *runtime.XType types.  These are the
	// discriminators on the runtime.Type interface.  We use local
	// reflection to fill these in from the remote symbol table,
	// so the names must match the runtime names.
	PBoolType,
	PUint8Type, PUint16Type, PUint32Type, PUint64Type, PUintType, PUintptrType,
	PInt8Type, PInt16Type, PInt32Type, PInt64Type, PIntType,
	PFloat32Type, PFloat64Type, PFloatType,
	PArrayType, PStringType, PStructType, PPtrType, PFuncType,
	PInterfaceType, PSliceType, PMapType, PChanType,
	PDotDotDotType, PUnsafePointerType proc.Word
	// G status values
	runtimeGStatus
}

// fillRuntimeIndexes fills a runtimeIndexes structure will the field
// indexes gathered from the remoteTypes recorded in a runtimeValues
// structure.
func fillRuntimeIndexes(runtime *runtimeValues, out *runtimeIndexes) {
	outv := reflect.Indirect(reflect.NewValue(out)).(*reflect.StructValue)
	outt := outv.Type().(*reflect.StructType)
	runtimev := reflect.Indirect(reflect.NewValue(runtime)).(*reflect.StructValue)

	// out contains fields corresponding to each runtime type
	for i := 0; i < outt.NumField(); i++ {
		// Find the interpreter type for this runtime type
		name := outt.Field(i).Name
		et := runtimev.FieldByName(name).Interface().(*remoteType).Type.(*eval.StructType)

		// Get the field indexes of the interpreter struct type
		indexes := make(map[string]int, len(et.Elems))
		for j, f := range et.Elems {
			if f.Anonymous {
				continue
			}
			name := f.Name
			if name[0] >= 'a' && name[0] <= 'z' {
				name = string(name[0]+'A'-'a') + name[1:]
			}
			indexes[name] = j
		}

		// Fill this field of out
		outStructv := outv.Field(i).(*reflect.StructValue)
		outStructt := outStructv.Type().(*reflect.StructType)
		for j := 0; j < outStructt.NumField(); j++ {
			f := outStructv.Field(j).(*reflect.IntValue)
			name := outStructt.Field(j).Name
			f.Set(int64(indexes[name]))
		}
	}
}
