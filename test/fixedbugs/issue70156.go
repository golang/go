// run

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"reflect"
)

func main() {
	pi := new(interface{})
	v := reflect.ValueOf(pi).Elem()
	if v.Kind() != reflect.Interface {
		panic(0)
	}
	if (v.Kind() == reflect.Ptr || v.Kind() == reflect.Interface) && v.IsNil() {
		return
	}
	panic(1)
}
