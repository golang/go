// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"./a"
	"reflect"
)

var s = []rune{0, 1, 2, 3}

func main() {
	m := map[any]int{}
	k := reflect.New(reflect.ArrayOf(4, reflect.TypeOf(int32(0)))).Elem().Interface()
	m[k] = 1
	a.F()
}
