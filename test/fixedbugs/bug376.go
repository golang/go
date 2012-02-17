// errorcheck

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 1951
package foo
import "unsafe"
var v = unsafe.Sizeof  // ERROR "must be called"

