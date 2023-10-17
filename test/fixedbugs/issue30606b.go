// run

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "reflect"

func main() {}

func typ(x interface{}) reflect.Type { return reflect.ValueOf(x).Type() }

var byteType = typ((byte)(0))
var ptrType = typ((*byte)(nil))

// Arrays of pointers. There are two size thresholds.
// Bit masks are chunked in groups of 120 pointers.
// Array types with >16384 pointers have a GC program instead of a bitmask.
var smallPtrType = reflect.ArrayOf(100, ptrType)
var mediumPtrType = reflect.ArrayOf(1000, ptrType)
var bigPtrType = reflect.ArrayOf(16385, ptrType)

var x0 = reflect.New(reflect.StructOf([]reflect.StructField{
	{Name: "F1", Type: byteType},
	{Name: "F2", Type: bigPtrType},
}))
var x1 = reflect.New(reflect.StructOf([]reflect.StructField{
	{Name: "F1", Type: smallPtrType},
	{Name: "F2", Type: bigPtrType},
}))
var x2 = reflect.New(reflect.StructOf([]reflect.StructField{
	{Name: "F1", Type: mediumPtrType},
	{Name: "F2", Type: bigPtrType},
}))
var x3 = reflect.New(reflect.StructOf([]reflect.StructField{
	{Name: "F1", Type: ptrType},
	{Name: "F2", Type: byteType},
	{Name: "F3", Type: bigPtrType},
}))
var x4 = reflect.New(reflect.StructOf([]reflect.StructField{
	{Name: "F1", Type: ptrType},
	{Name: "F2", Type: smallPtrType},
	{Name: "F3", Type: bigPtrType},
}))
var x5 = reflect.New(reflect.StructOf([]reflect.StructField{
	{Name: "F1", Type: ptrType},
	{Name: "F2", Type: mediumPtrType},
	{Name: "F3", Type: bigPtrType},
}))
