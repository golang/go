// compile

// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 6847: select clauses involving implicit conversion
// of channels trigger a spurious typechecking error during walk.

package p

type I1 interface {
	String()
}
type I2 interface {
	String()
}

func F() {
	var (
		cr <-chan int
		cs chan<- int
		c  chan int

		ccr chan (<-chan int)
		ccs chan chan<- int
		cc  chan chan int

		ok bool
	)
	// Send cases.
	select {
	case ccr <- cr:
	case ccr <- c:
	}
	select {
	case ccs <- cs:
	case ccs <- c:
	}
	select {
	case ccr <- c:
	default:
	}
	// Receive cases.
	select {
	case cr = <-cc:
	case cs = <-cc:
	case c = <-cc:
	}
	select {
	case cr = <-cc:
	default:
	}
	select {
	case cr, ok = <-cc:
	case cs, ok = <-cc:
	case c = <-cc:
	}
      // Interfaces.
	var (
		c1 chan I1
		c2 chan I2
		x1 I1
		x2 I2
	)
	select {
	case c1 <- x1:
	case c1 <- x2:
	case c2 <- x1:
	case c2 <- x2:
	}
	select {
	case x1 = <-c1:
	case x1 = <-c2:
	case x2 = <-c1:
	case x2 = <-c2:
	}
	select {
	case x1, ok = <-c1:
	case x1, ok = <-c2:
	case x2, ok = <-c1:
	case x2, ok = <-c2:
	}
	_ = ok
}
