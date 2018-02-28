// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Fail to guess the kind of the constant "x".
// No runtime test; just make sure it compiles.

package cgotest

// const int x = 42;
import "C"

var issue21668_X = C.x
