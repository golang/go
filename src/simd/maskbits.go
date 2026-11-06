// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package simd

// MaxMaskBitsLen is the maximum number of uint words written by a mask's
// ToBits method. An array of this length provides sufficient storage for
// any mask type, independent of the selected vector length.
//
// This constant may increase as support for wider vectors is added.
const MaxMaskBitsLen = 1
