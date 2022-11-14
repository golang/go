// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package b

import "a"

// ----------------------------------------------------------------------------
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

// Associated with comparable type if AllDecls is set.
func ComparableFactory() comparable {}

// Should only appear if AllDecls is set.
type uint struct{} // overrides a predeclared type uint

// Should only appear if AllDecls is set.
type comparable struct{} // overrides a predeclared type comparable

// ----------------------------------------------------------------------------
// Exported declarations associated with non-exported types must always be shown.

type notExported int

const C notExported = 0

const (
	C1 notExported = iota
	C2
	c3
	C4
	C5
)

var V notExported
var V1, V2, v3, V4, V5 notExported

var (
	U1, U2, u3, U4, U5 notExported
	u6                 notExported
	U7                 notExported = 7
)

func F1() notExported {}
func f2() notExported {}
