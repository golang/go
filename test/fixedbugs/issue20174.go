// compile -c=2

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 20174: failure to typecheck contents of *T in the frontend.

package p

func f() {
	_ = (*interface{})(nil) // interface{} here used to not have its width calculated going into backend
	select {
	case _ = <-make(chan interface {
		M()
	}, 1):
	}
}
