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

func TestIsRegularMemory(t *testing.T) {
	type args struct {
		t reflect.Type
	}
	type S struct {
		int
	}
	tests := []struct {
		name string
		args args
		want bool
	}{
		{"struct{i int}", args{reflect.TypeOf(struct{ i int }{})}, true},
		{"struct{}", args{reflect.TypeOf(struct{}{})}, true},
		{"struct{i int; s S}", args{reflect.TypeOf(struct {
			i int
			s S
		}{})}, true},
		{"map[int][int]", args{reflect.TypeOf(map[int]int{})}, false},
		{"[4]chan int", args{reflect.TypeOf([4]chan int{})}, true},
		{"struct{i int; _ S}", args{reflect.TypeOf(struct {
			i int
			_ S
		}{})}, false},
		{"struct{a int16; b int32}", args{reflect.TypeOf(struct {
			a int16
			b int32
		}{})}, false},
		{"struct { x int32; y int16}", args{reflect.TypeOf(struct {
			x int32
			y int16
		}{})}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := reflect.IsRegularMemory(tt.args.t); got != tt.want {
				t.Errorf("isRegularMemory() = %v, want %v", got, tt.want)
			}
		})
	}
}
