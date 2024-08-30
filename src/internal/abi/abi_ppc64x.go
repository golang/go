// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ppc64 || ppc64le

package abi

const (
	// See abi_generic.go.

	// R3 - R10, R14 - R17.
	IntArgRegs = 12

	// F1 - F12.
	FloatArgRegs = 12

	EffectiveFloatRegSize = 8
)
