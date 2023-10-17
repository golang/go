// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "reflect"

func main() {
	_ = reflect.StructOf([]reflect.StructField{
		{Name: "_", PkgPath: "main", Type: reflect.TypeOf(int(0))},
		{Name: "_", PkgPath: "main", Type: reflect.TypeOf(int(0))},
	})
}
