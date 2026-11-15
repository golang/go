// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt

import (
	"reflect"
	"strconv"
	"unsafe"
)

// ViewAs reinterprets the leading bytes of b as a *T (SPEC T-205,
// M7-S5). T must be a POD struct: the compiler rejects padding and
// reference-carrying fields with [comptime::viewas_not_pod]; the runtime
// mirrors that contract here so an untrusted Go dependency calling
// logos/rt directly cannot forge slice headers or other references.
// A short input panics with `logos: ViewAs: need <N> bytes, have <M>`;
// a non-POD T panics with a distinct `logos: ViewAs:` message.
//
// The returned pointer addresses b's backing array, so a live view
// keeps b alive through the collector's interior-pointer resolution;
// on architectures without unaligned loads the view addresses an
// aligned copy instead (viewas_copy.go).
func ViewAs[T any](b []byte) *T {
	viewAsEnforcePOD[T]()
	var zero T
	size := int(unsafe.Sizeof(zero))
	if len(b) < size {
		panic("logos: ViewAs: need " + strconv.Itoa(size) + " bytes, have " + strconv.Itoa(len(b)))
	}
	return viewAsAddr[T](b)
}

func viewAsEnforcePOD[T any]() {
	t := reflect.TypeFor[T]()
	if t.Kind() != reflect.Struct {
		panic("logos: ViewAs: type " + t.String() + " is not a struct")
	}
	if off, kind := viewAsScan(t, 0); kind != "" {
		panic("logos: ViewAs: type " + t.String() + " is not POD: " + kind + " at offset " + strconv.FormatUint(uint64(off), 10))
	}
}

func viewAsScan(t reflect.Type, base uintptr) (uintptr, string) {
	switch t.Kind() {
	case reflect.Struct:
		prev := base
		for i := 0; i < t.NumField(); i++ {
			f := t.Field(i)
			abs := base + f.Offset
			if abs > prev {
				return prev, "padding"
			}
			if off, kind := viewAsScan(f.Type, abs); kind != "" {
				return off, kind
			}
			prev = abs + f.Type.Size()
		}
		if base+t.Size() > prev {
			return prev, "padding"
		}
		return 0, ""
	case reflect.Array:
		return viewAsScan(t.Elem(), base)
	case reflect.Bool:
		return base, "bool"
	case reflect.String:
		return base, "string"
	case reflect.Slice:
		return base, "slice"
	case reflect.Map:
		return base, "map"
	case reflect.Interface, reflect.Pointer, reflect.UnsafePointer:
		return base, "pointer"
	case reflect.Chan:
		return base, "channel"
	case reflect.Func:
		return base, "func"
	}
	return 0, ""
}

func viewAsCopy[T any](b []byte) *T {
	var zero T
	size := int(unsafe.Sizeof(zero))
	p := new(T)
	if size > 0 {
		copy(unsafe.Slice((*byte)(unsafe.Pointer(p)), size), b)
	}
	return p
}
