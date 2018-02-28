// errorcheck

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

const _ = 6e5518446744 // ERROR "malformed constant: 6e5518446744 \(exponent overflow\)"
const _ = 1e-1000000000
const _ = 1e+1000000000 // ERROR "constant too large"
