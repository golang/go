// $G $D/$F.go || echo BUG: bug316

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 1369.

package main

const (
	c = complex(1, 2)
	r = real(c) // was: const initializer must be constant
	i = imag(c) // was: const initializer must be constant
)

func main() {}
