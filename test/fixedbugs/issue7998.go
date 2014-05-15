// compile

// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// /tmp/x.go:5: cannot use _ as value

package p

func f(ch chan int) bool {
	select {
	case _, ok := <-ch:
		return ok
	}
	_, ok := <-ch
	_ = ok
	select {
	case _, _ = <-ch:
		return true
	}
	return false
}
