// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Some additional checks for inferred composite literal types

package comp_literals

type S1 struct{
       x int
       y float64
       s string
}

var (
       _ = { /* ERROR "missing type in composite literal" */ }
       _ []int = {}
       _ struct{} = {}
      _ [10]byte = {1, 2, 3, 9: 10}
       _ map[string]int = {"foo": 0, "bar": 1}
)

// var (
//        _ = struct{ f struct { f int }}{{1}}
// )

func _() []int {
       return nil
       return {}
       return {1, 2, 3}
}

func _() S1 {
       return {}
       return {1, 2, "3"}
       return {1.0, 2, ""}
       return {s: "foo"}
}

func _() []S1 {
       return {}
       return {{}}
       return {{x: 1}, {y: 2}, {1, 2.0, "3"}}
}

func f(s []int) {
       s = {}
       s = {1, 2, 3}
       s = {0: 0, 1: 1}
       s = {"foo" /* ERRORx "cannot use .* as int value" */ }
}

// func _() {
//        f({})
// }

type S2 struct {
       f func(x int)
}

func g1[T any](x T) {}

var _ S2 = { g1 }
