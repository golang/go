// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmath

// Conj returns the complex conjugate of x.
func Conj(x complex128) complex128 { return complex(real(x), -imag(x)) }
