// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Composite literals with inferred types

package comp_literals

type T struct {}

// var declaration
func _() {
        var _ T = {}
}

// assignment
func _() {
        var x T
        x = {}
        _ = x // "use" it
}

// // argument to function call
// func _() {
//         f := func(_ T) {}
//         f({})
// }

// // argument to function call with variadic arguments
// func _() {
//         f := func(_ ...T) {}
//         f({}, {})
// }

// map index expression
func _() {
        var x map[T]int
        _ = x[{}]
}

// map index expression through a type parameter
func _[P map[T]int](x P) {
        _ = x[{}]
}

// // value of a struct literal with keys / values
// func _() {
//         type S struct {
//                 f T
//         }
//         _ = S{f: {}}
// }

// // value of a struct literal without keys / values
// func _() {
//         type S struct {
//                 f T
//         }
//         _ = S{{}}
// }

// values sent to a channel
func _() {
        var x chan<- T
        x <- {}
}

// argument to conversion
func _() {
        _ = T({})
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
