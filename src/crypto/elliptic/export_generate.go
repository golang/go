// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build tablegen

package elliptic

// This block exports p256-related internals for the p256 table generator in internal/gen.
var (
	P256PointDoubleAsm = p256PointDoubleAsm
	P256PointAddAsm    = p256PointAddAsm
	P256Inverse        = p256Inverse
	P256Sqr            = p256Sqr
	P256Mul            = p256Mul
)
