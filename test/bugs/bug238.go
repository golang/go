// errchk $G -e $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test case for issue 471. This file shouldn't compile.

package main

const a *int = 1        // ERROR "wrong|incompatible"
const b [2]int = 2      // ERROR "wrong|incompatible"
const c map[int]int = 3 // ERROR "wrong|incompatible"
const d chan int = 4    // ERROR "wrong|incompatible"
const e func() = 5      // ERROR "wrong|incompatible"
const f struct{} = 6    // ERROR "wrong|incompatible"
const g interface{} = 7 // ERROR "wrong|incompatible"

func main() { println(a, b, c, d, e, f, g) }
