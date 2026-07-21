// -lang=go1.27

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This is a replica of compliterals1.go at Go 1.27 to ensure error
// messages are reported as expected.

package comp_literals

type T struct {}

// var declaration
func _() {
        var _ T = { /* ERROR "requires go1.28 or later" */ }
}

// assignment
func _() {
        var x T
        x = { /* ERROR "requires go1.28 or later" */ }
        _ = x // "use" it
}

// // argument to function call
// func _() {
//         f := func(_ T) {}
//         f({ /* ERROR "requires go1.28 or later" */ })
// }

// // argument to function call with variadic arguments
// func _() {
//         f := func(_ ...T) {}
//         f({ /* ERROR "requires go1.28 or later" */ }, { /* ERROR "requires go1.28 or later" */ })
// }

// map index expression
func _() {
        var x map[T]int
        _ = x[{ /* ERROR "requires go1.28 or later" */ }]
}

// map index expression through a type parameter
func _[P map[T]int](x P) {
        _ = x[{ /* ERROR "requires go1.28 or later" */ }]
}

// // value of a struct literal with keys / values
// func _() {
//         type S struct {
//                 f T
//         }
//         _ = S{f: { /* ERROR "requires go1.28 or later" */ }}
// }

// // value of a struct literal without keys / values
// func _() {
//         type S struct {
//                 f T
//         }
//         _ = S{{ /* ERROR "requires go1.28 or later" */ }}
// }

// values sent to a channel
func _() {
        var x chan<- T
        x <- { /* ERROR "requires go1.28 or later" */ }
}

// argument to conversion
func _() {
        _ = T({ /* ERROR "requires go1.28 or later" */ })
}

// The below are all covered by the inference mechanism used before
// Go 1.28 (AKA "hints"). They are included here for completeness.

// keys of a map literal
func _() {
        type M map[T]int
        _ = M{{}: 42}
}

// values of a map literal
func _() {
        type M map[int]T
        _ = M{42: {}}
}

// elements in an array literal
func _() {
        _ = [42]T{{}}
}

// elements in a slice literal
func _() {
        _ = []T{{}}
}
