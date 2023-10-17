// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

func f[T any](i interface{}) {
	switch x := i.(type) {
	case T:
		fmt.Println("T", x)
	case int:
		fmt.Println("int", x)
	case int32, int16:
		fmt.Println("int32/int16", x)
	case struct{ a, b T }:
		fmt.Println("struct{T,T}", x.a, x.b)
	default:
		fmt.Println("other", x)
	}
}
func main() {
	f[float64](float64(6))
	f[float64](int(7))
	f[float64](int32(8))
	f[float64](struct{ a, b float64 }{a: 1, b: 2})
	f[float64](int8(9))
	f[int32](int32(7))
	f[int](int32(7))
	f[any](int(10))
	f[interface{ M() }](int(11))
}
