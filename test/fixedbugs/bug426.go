// compile

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// gccgo crashed compiling this.

package p

type T *T

func f(t T) {
	println(t, *t)
}
