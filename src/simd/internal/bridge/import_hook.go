// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package bridge

// ZeroSized is used as the definition for type _simd in package simd, to create
// a hard dependence between the two packages before code transformation.
type ZeroSized struct {
	_ [0]func(*ZeroSized) *ZeroSized
}
