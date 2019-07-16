// compile

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 4162. Trailing commas now allowed in conversions.

package p

// All these are valid now.
var (
	_ = int(1.0,)      // comma was always permitted (like function call)
	_ = []byte("foo",) // was syntax error: unexpected comma
	_ = chan int(nil,) // was syntax error: unexpected comma
	_ = (func())(nil,) // was syntax error: unexpected comma
)
