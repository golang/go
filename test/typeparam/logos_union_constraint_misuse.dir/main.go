// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "./a"

func ordered[T a.Ordered]() {}

func number[T a.Number]() {}

func labeled[T a.Labeled]() {}

type named int

func (named) Label() string { return "" }

type myInt int

func main() {
	ordered[int]()
	ordered[string]()
	ordered[myInt]()
	ordered[bool]()     // ERROR "bool does not satisfy a.Ordered"
	ordered[struct{}]() // ERROR "struct{} does not satisfy a.Ordered"
	ordered[[]int]()    // ERROR "\[\]int does not satisfy a.Ordered"

	number[int]()
	number[float64]()
	number[string]() // ERROR "string does not satisfy a.Number"
	number[myInt]()  // ERROR "myInt does not satisfy a.Number"

	labeled[named]()
	labeled[int]()    // ERROR "int does not satisfy a.Labeled"
	labeled[myInt]()  // ERROR "myInt does not satisfy a.Labeled"
	labeled[string]() // ERROR "string does not satisfy a.Labeled"
}
