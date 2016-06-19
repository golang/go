// compile

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 13821.  Compiler rejected "bool(true)" as not a constant.

package p

const (
	A = true
	B = bool(A)
	C = bool(true)
)
