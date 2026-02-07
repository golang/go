// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flate

const (
	// Masks for shifts with register sizes of the shift value.
	// This can be used to work around the x86 design of shifting by mod register size.
	// It can be used when a variable shift is always smaller than the register size.

	// reg8SizeMask64 - shift value is 8 bits on 64 bit register.
	reg8SizeMask64 = 63
)
