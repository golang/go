// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package b

import "a"

// Basic declarations

const Pi = 3.14   // Pi
var MaxInt int    // MaxInt
type T struct{}   // T
var V T           // v
func F(x int) int {} // F
func (x *T) M()   {} // M

// Corner cases: association with (presumed) predeclared types

// Always under the package functions list.
func NotAFactory() int {}

// Associated with uint type if AllDecls is set.
func UintFactory() uint {}

// Associated with uint type if AllDecls is set.
func uintFactory() uint {}

// Should only appear if AllDecls is set.
type uint struct{} // overrides a predeclared type uint
