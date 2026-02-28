// run

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "reflect"

func main() {}

func typ(x interface{}) reflect.Type { return reflect.ValueOf(x).Type() }

var x = reflect.New(reflect.StructOf([]reflect.StructField{
	{Name: "F5", Type: reflect.StructOf([]reflect.StructField{
		{Name: "F4", Type: reflect.ArrayOf(5462,
			reflect.SliceOf(typ(uint64(0))))},
	})},
}))
