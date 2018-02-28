// errorcheck

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 20227: panic while constructing constant "1i/1e-600000000"

package p

var _ = 1 / 1e-600000000i  // ERROR "complex division by zero"
var _ = 1i / 1e-600000000  // ERROR "complex division by zero"
var _ = 1i / 1e-600000000i // ERROR "complex division by zero"

var _ = 1 / (1e-600000000 + 1e-600000000i)  // ERROR "complex division by zero"
var _ = 1i / (1e-600000000 + 1e-600000000i) // ERROR "complex division by zero"
