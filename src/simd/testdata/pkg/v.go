// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simdstable

package pkg

// For testing purposes, F and V are exported simd types,
// and should have the proper (variable) unsafe.Sizeof

import (
	"simd"
)

var V simd.Float32s

// unexportedAlias and UnexportedAlias let another package initialize a
// package-level variable whose inferred type is an unexported alias from pkg.
type unexportedAlias = int

func UnexportedAlias(x int) unexportedAlias {
	return x
}

func F() simd.Float32s {
	var x simd.Float32s
	return x
}
