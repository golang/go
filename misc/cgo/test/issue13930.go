// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 13930.  Test that cgo's multiple-value special form for
// C function calls works in variable declaration statements.

package cgotest

// #include <stdlib.h>
import "C"

var _, _ = C.abs(0)
