// run

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The PkgPath of unexported fields of types defined in package main was incorrectly ""

package main

import (
	"fmt"
	"reflect"
)

type foo struct {
	bar int
}

func main() {
	pkgpath := reflect.ValueOf(foo{}).Type().Field(0).PkgPath
	if pkgpath != "main" {
		fmt.Printf("BUG: incorrect PkgPath: %v", pkgpath)
	}
}
