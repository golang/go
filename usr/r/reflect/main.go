// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"reflect"
)

func main() {
	var s string;
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

	t = reflect.ParseTypeString("*chan<-string");
	s = reflect.ToString(t); print(s, "\n");

	t = reflect.ParseTypeString("struct {c *chan *int32; d float32}");
	s = reflect.ToString(t); print(s, "\n");

	t = reflect.ParseTypeString("*(a int8, b int32)");
	s = reflect.ToString(t); print(s, "\n");

	t = reflect.ParseTypeString("struct {c *(? *chan *int32, ? *int8)}");
	s = reflect.ToString(t); print(s, "\n");
}
