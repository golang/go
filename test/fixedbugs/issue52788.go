// run

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 52788: miscompilation for boolean comparison on ARM64.

package main

import (
	"fmt"
	"reflect"
)

func f(next func() bool) {
	for b := next(); b; b = next() {
		fmt.Printf("next() returned %v\n", b)
	}
}

func main() {
	next := reflect.MakeFunc(reflect.TypeOf((func() bool)(nil)), func(_ []reflect.Value) []reflect.Value {
		return []reflect.Value{reflect.ValueOf(false)}
	})
	reflect.ValueOf(f).Call([]reflect.Value{next})
}
