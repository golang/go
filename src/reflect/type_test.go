// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package reflect_test

import (
	"reflect"
	"testing"
)

func TestTypeFor(t *testing.T) {
	type (
		mystring string
		myiface  interface{}
	)

	testcases := []struct {
		wantFrom any
		got      reflect.Type
	}{
		{new(int), reflect.TypeFor[int]()},
		{new(int64), reflect.TypeFor[int64]()},
		{new(string), reflect.TypeFor[string]()},
		{new(mystring), reflect.TypeFor[mystring]()},
		{new(any), reflect.TypeFor[any]()},
		{new(myiface), reflect.TypeFor[myiface]()},
	}
	for _, tc := range testcases {
		want := reflect.ValueOf(tc.wantFrom).Elem().Type()
		if want != tc.got {
			t.Errorf("unexpected reflect.Type: got %v; want %v", tc.got, want)
		}
	}
}

func TestStructOfEmbeddedIfaceMethodCall(t *testing.T) {
	type Named interface {
		Name() string
	}

	typ := reflect.StructOf([]reflect.StructField{
		{
			Anonymous: true,
			Name:      "Named",
			Type:      reflect.TypeFor[Named](),
		},
	})

	v := reflect.New(typ).Elem()
	v.Field(0).Set(
		reflect.ValueOf(reflect.TypeFor[string]()),
	)

	x := v.Interface().(Named)
	shouldPanic("StructOf does not support methods of embedded interfaces", func() {
		_ = x.Name()
	})
}
