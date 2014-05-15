// compile

// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// /tmp/x.go:3: internal error: f &p (type *int) recorded as live on entry

package p

func f(ch chan int) *int {
	select {
	case p1x := <-ch:
		return &p1x
	default:
		// ok
	}
	select {
	case p1 := <-ch:
		return &p1
	default:
		// ok
	}
	select {
	case p2 := <-ch:
		return &p2
	case p3 := <-ch:
		return &p3
	default:
		// ok
	}
	select {
	case p4, ok := <-ch:
		if ok {
			return &p4
		}
	default:
		// ok
	}
	select {
	case p5, ok := <-ch:
		if ok {
			return &p5
		}
	case p6, ok := <-ch:
		if !ok {
			return &p6
		}
	default:
		// ok
	}
	return nil
}
