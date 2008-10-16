// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"reflect"
)

func main() {
	var s string;

	if false {
		s = reflect.ToString(reflect.Int8); print(s, "\n");
		s = reflect.ToString(reflect.Int16); print(s, "\n");
		s = reflect.ToString(reflect.Int32); print(s, "\n");
		s = reflect.ToString(reflect.Int64); print(s, "\n");
		s = reflect.ToString(reflect.Uint8); print(s, "\n");
		s = reflect.ToString(reflect.Uint16); print(s, "\n");
		s = reflect.ToString(reflect.Uint32); print(s, "\n");
		s = reflect.ToString(reflect.Uint64); print(s, "\n");
		s = reflect.ToString(reflect.Float32); print(s, "\n");
		s = reflect.ToString(reflect.Float64); print(s, "\n");
		s = reflect.ToString(reflect.Float80); print(s, "\n");
		s = reflect.ToString(reflect.String); print(s, "\n");

		s = reflect.ToString(reflect.PtrInt8); print(s, "\n");
		s = reflect.ToString(reflect.PtrPtrInt8); print(s, "\n");
		s = reflect.ToString(reflect.ArrayFloat32); print(s, "\n");
		s = reflect.ToString(reflect.MapStringInt16); print(s, "\n");
		s = reflect.ToString(reflect.ChanArray); print(s, "\n");
		s = reflect.ToString(reflect.Structure); print(s, "\n");
		s = reflect.ToString(reflect.Function); print(s, "\n");
	}
	var t reflect.Type;

	t = reflect.ParseTypeString("int8");
	s = reflect.ToString(t); print(s, "\n");

	t = reflect.ParseTypeString("**int8");
	s = reflect.ToString(t); print(s, "\n");

	t = reflect.ParseTypeString("**P.integer");
	s = reflect.ToString(t); print(s, "\n");

	t = reflect.ParseTypeString("[32]int32");
	s = reflect.ToString(t); print(s, "\n");

	t = reflect.ParseTypeString("[]int8");
	s = reflect.ToString(t); print(s, "\n");

	t = reflect.ParseTypeString("map[string]int32");
	s = reflect.ToString(t); print(s, "\n");
}
