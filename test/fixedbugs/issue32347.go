// compile

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 32347: gccgo compiler crashes with int-to-string conversion
// with large integer constant operand.

package p

const (
	X1 = string(128049)
	X2 = string(-1)
	X3 = string(1<<48)
)

var S1, S2, S3 = X1, X2, X3
