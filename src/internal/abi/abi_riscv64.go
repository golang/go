// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.regabiargs

package abi

const (
	// See abi_generic.go.

	// X8 - X23
	IntArgRegs = 16

	// F8 - F23.
	FloatArgRegs = 16

	EffectiveFloatRegSize = 8
)
