// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unsafeheader_test

import (
	"bytes"
	"internal/unsafeheader"
	"reflect"
	"testing"
	"unsafe"
)

// TestTypeMatchesReflectType ensures that the name and layout of the
// unsafeheader types matches the corresponding Header types in the reflect
// package.
func TestTypeMatchesReflectType(t *testing.T) {
	t.Run("Slice", func(t *testing.T) {
		testHeaderMatchesReflect(t, unsafeheader.Slice{}, reflect.SliceHeader{})
	})

	t.Run("String", func(t *testing.T) {
		testHeaderMatchesReflect(t, unsafeheader.String{}, reflect.StringHeader{})
	})
}

func testHeaderMatchesReflect(t *testing.T, header, reflectHeader interface{}) {
	h := reflect.TypeOf(header)
	rh := reflect.TypeOf(reflectHeader)

	for i := 0; i < h.NumField(); i++ {
		f := h.Field(i)
		rf, ok := rh.FieldByName(f.Name)
		if !ok {
			t.Errorf("Field %d of %v is named %s, but no such field exists in %v", i, h, f.Name, rh)
			continue
		}
		if !typeCompatible(f.Type, rf.Type) {
			t.Errorf("%v.%s has type %v, but %v.%s has type %v", h, f.Name, f.Type, rh, rf.Name, rf.Type)
		}
		if f.Offset != rf.Offset {
			t.Errorf("%v.%s has offset %d, but %v.%s has offset %d", h, f.Name, f.Offset, rh, rf.Name, rf.Offset)
		}
	}

	if h.NumField() != rh.NumField() {
		t.Errorf("%v has %d fields, but %v has %d", h, h.NumField(), rh, rh.NumField())
	}
	if h.Align() != rh.Align() {
		t.Errorf("%v has alignment %d, but %v has alignment %d", h, h.Align(), rh, rh.Align())
	}
}

var (
	unsafePointerType = reflect.TypeOf(unsafe.Pointer(nil))
	uintptrType       = reflect.TypeOf(uintptr(0))
)

func typeCompatible(t, rt reflect.Type) bool {
	return t == rt || (t == unsafePointerType && rt == uintptrType)
}

// TestWriteThroughHeader ensures that the headers in the unsafeheader package
// can successfully mutate variables of the corresponding built-in types.
//
// This test is expected to fail under -race (which implicitly enables
// -d=checkptr) if the runtime views the header types as incompatible with the
// underlying built-in types.
func TestWriteThroughHeader(t *testing.T) {
	t.Run("Slice", func(t *testing.T) {
		s := []byte("Hello, checkptr!")[:5]

		var alias []byte
		hdr := (*unsafeheader.Slice)(unsafe.Pointer(&alias))
		hdr.Data = unsafe.Pointer(&s[0])
		hdr.Cap = cap(s)
		hdr.Len = len(s)

		if !bytes.Equal(alias, s) {
			t.Errorf("alias of %T(%q) constructed via Slice = %T(%q)", s, s, alias, alias)
		}
		if cap(alias) != cap(s) {
			t.Errorf("alias of %T with cap %d has cap %d", s, cap(s), cap(alias))
		}
	})

	t.Run("String", func(t *testing.T) {
		s := "Hello, checkptr!"

		var alias string
		hdr := (*unsafeheader.String)(unsafe.Pointer(&alias))
		hdr.Data = (*unsafeheader.String)(unsafe.Pointer(&s)).Data
		hdr.Len = len(s)

		if alias != s {
			t.Errorf("alias of %q constructed via String = %q", s, alias)
		}
	})
}
