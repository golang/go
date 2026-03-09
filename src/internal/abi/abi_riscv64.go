// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package abi

const (
	// See abi_generic.go.

	// X10 - X17, X9, X18 - X23.
	IntArgRegs = 15

	// F8 - F23.
	FloatArgRegs = 16

	EffectiveFloatRegSize = 8
)
