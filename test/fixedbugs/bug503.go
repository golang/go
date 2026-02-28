// compile

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// gccgo crashed compiling this file, due to failing to correctly emit
// the type descriptor for a named alias.

package p

type entry = struct {
	a, b, c int
}

var V entry
