// run

// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 7363: CanSet must return false for unexported embedded struct fields.

package main

import "reflect"

type a struct {
}

type B struct {
	a
}

func main() {
	b := &B{}
	v := reflect.ValueOf(b).Elem().Field(0)
	if v.CanSet() {
		panic("B.a is an unexported embedded struct field")
	}
}
