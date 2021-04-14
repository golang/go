// run

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// We were picking up a special noalg type from typelinks.

package main

import "reflect"

func f(m map[string]int) int {
	return m["a"]
}

func g(m map[[8]string]int) int {
	t := reflect.ArrayOf(8, reflect.TypeOf(""))
	a := reflect.New(t).Elem()
	return m[a.Interface().([8]string)]
}

func main() {
	m := map[[8]string]int{}
	g(m)
}
