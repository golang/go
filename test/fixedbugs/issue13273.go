// errorcheck

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check that we correctly construct (and report errors)
// for unary expressions of the form <-x where we only
// know after parsing x whether <-x is a receive operation
// or a channel type.

package n

func f() {
	// test case from issue 13273
	<-chan int((chan int)(nil))

	<-chan int(nil)
	<-chan chan int(nil)
	<-chan chan chan int(nil)
	<-chan chan chan chan int(nil)
	<-chan chan chan chan chan int(nil)

	<-chan<-chan int(nil)
	<-chan<-chan<-chan int(nil)
	<-chan<-chan<-chan<-chan int(nil)
	<-chan<-chan<-chan<-chan<-chan int(nil)

	<-chan (<-chan int)(nil)
	<-chan (<-chan (<-chan int))(nil)
	<-chan (<-chan (<-chan (<-chan int)))(nil)
	<-chan (<-chan (<-chan (<-chan (<-chan int))))(nil)

	<-(<-chan int)(nil)
	<-(<-chan chan int)(nil)
	<-(<-chan chan chan int)(nil)
	<-(<-chan chan chan chan int)(nil)
	<-(<-chan chan chan chan chan int)(nil)

	<-(<-chan<-chan int)(nil)
	<-(<-chan<-chan<-chan int)(nil)
	<-(<-chan<-chan<-chan<-chan int)(nil)
	<-(<-chan<-chan<-chan<-chan<-chan int)(nil)

	<-(<-chan (<-chan int))(nil)
	<-(<-chan (<-chan (<-chan int)))(nil)
	<-(<-chan (<-chan (<-chan (<-chan int))))(nil)
	<-(<-chan (<-chan (<-chan (<-chan (<-chan int)))))(nil)

	type _ <-<-chan int // ERROR "unexpected <-, expected chan|expected .*chan.*"
	<-<-chan int // ERROR "unexpected <-, expected chan|expecting {" (new parser: same error as for type decl)

	type _ <-chan<-int // ERROR "unexpected int, expected chan|expected .*chan.*|expected chan|expected .*;.* or .*}.* or newline"
	<-chan<-int // ERROR "unexpected int, expected chan|expecting {" (new parser: same error as for type decl)
}
