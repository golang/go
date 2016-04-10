// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 8331.  A typedef of an unnamed struct is the same struct when
// #include'd twice.  No runtime test; just make sure it compiles.

package cgotest

// #include "issue8331.h"
import "C"

func issue8331a() C.issue8331 {
	return issue8331Var
}
