// errorcheck -d=panic

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

const x = 6e5518446744 // ERROR "malformed constant: 6e5518446744"
const _ = x * x
const _ = 1e-1000000000
const _ = 1e+1000000000 // ERROR "malformed constant: 1e\+1000000000"
