// errorcheck

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 11674: cmd/compile: does not diagnose constant division by
// zero

package p

const x complex64 = 0
const y complex128 = 0

var _ = x / 1e-20
var _ = x / 1e-50   // GC_ERROR "division by zero"
var _ = x / 1e-1000 // GC_ERROR "division by zero"
var _ = x / 1e-20i
var _ = x / 1e-50i   // GC_ERROR "division by zero"
var _ = x / 1e-1000i // GC_ERROR "division by zero"

var _ = x / 1e-45 // smallest positive float32

var _ = x / (1e-20 + 1e-20i)
var _ = x / (1e-50 + 1e-20i)
var _ = x / (1e-20 + 1e-50i)
var _ = x / (1e-50 + 1e-50i)     // GC_ERROR "division by zero"
var _ = x / (1e-1000 + 1e-1000i) // GC_ERROR "division by zero"

var _ = y / 1e-50
var _ = y / 1e-1000 // GC_ERROR "division by zero"
var _ = y / 1e-50i
var _ = y / 1e-1000i // GC_ERROR "division by zero"

var _ = y / 5e-324 // smallest positive float64

var _ = y / (1e-50 + 1e-50)
var _ = y / (1e-1000 + 1e-50i)
var _ = y / (1e-50 + 1e-1000i)
var _ = y / (1e-1000 + 1e-1000i) // GC_ERROR "division by zero"
