// compile

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T interface {
	M(P)
}

type M interface {
	F() P
}

type P = interface {
	// The compiler cannot handle this case. Disabled for now.
	// See issue #25838.
	// I() M
}

func main() {}
