// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f() {}

const c = 0

var v int
var _ = f < c // ERROR "invalid operation: f < c (mismatched types func() and untyped int)"
var _ = f < v // ERROR "invalid operation: f < v (mismatched types func() and int)"
