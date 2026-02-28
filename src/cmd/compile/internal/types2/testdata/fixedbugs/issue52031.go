// -lang=go1.12

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type resultFlags uint

// Example from #52031.
//
// The following shifts should not produce errors on Go < 1.13, as their
// untyped constant operands are representable by type uint.
const (
	_ resultFlags = (1 << iota) / 2

	reportEqual
	reportUnequal
	reportByIgnore
	reportByMethod
	reportByFunc
	reportByCycle
)

// Invalid cases.
var x int = 1
var _ = (8 << x /* ERROR "signed shift count .* requires go1.13 or later" */)

const _ = (1 << 1.2 /* ERROR "truncated to uint" */)

var y float64
var _ = (1 << y /* ERROR "must be integer" */)
