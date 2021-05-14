// errorcheck

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// No double error on ideal -> float{32,64} conversion overflow

package issue19947

var _ = float32(1) * 1e200 // ERROR "constant 1e\+200 overflows float32|1e200 .* overflows float32"
var _ = float64(1) * 1e500 // ERROR "constant 1e\+500 overflows float64|1e500 .* overflows float64"

var _ = complex64(1) * 1e200  // ERROR "constant 1e\+200 overflows complex64|1e200 .* overflows complex64"
var _ = complex128(1) * 1e500 // ERROR "constant 1e\+500 overflows complex128|1e500 .* overflows complex128"
