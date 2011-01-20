// errchk $G -e $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test case for issue 471. This file shouldn't compile.

package main

const a *int = 1        // ERROR "convert|wrong|invalid"
const b [2]int = 2      // ERROR "convert|wrong|invalid"
const c map[int]int = 3 // ERROR "convert|wrong|invalid"
const d chan int = 4    // ERROR "convert|wrong|invalid"
const e func() = 5      // ERROR "convert|wrong|invalid"
const f struct{} = 6    // ERROR "convert|wrong|invalid"
const g interface{} = 7 // ERROR "constant|wrong|invalid"
const h bool = false
const i int = 2
const j float64 = 5

func main() { println(a, b, c, d, e, f, g) }
