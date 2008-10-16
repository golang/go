// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"reflect"
)

func main() {
	reflect.Print(reflect.Int8); print("\n");
	reflect.Print(reflect.Int16); print("\n");
	reflect.Print(reflect.Int32); print("\n");
	reflect.Print(reflect.Int64); print("\n");
	reflect.Print(reflect.Uint8); print("\n");
	reflect.Print(reflect.Uint16); print("\n");
	reflect.Print(reflect.Uint32); print("\n");
	reflect.Print(reflect.Uint64); print("\n");
	reflect.Print(reflect.Float32); print("\n");
	reflect.Print(reflect.Float64); print("\n");
	reflect.Print(reflect.Float80); print("\n");
	reflect.Print(reflect.String); print("\n");

	reflect.Print(reflect.PtrInt8); print("\n");
	reflect.Print(reflect.ArrayFloat32); print("\n");
	reflect.Print(reflect.MapStringInt16); print("\n");
	reflect.Print(reflect.ChanArray); print("\n");
	reflect.Print(reflect.Structure); print("\n");
	reflect.Print(reflect.Function); print("\n");
}
