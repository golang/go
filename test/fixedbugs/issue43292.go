// run

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	{
		i := I(A{})

		b := make(chan I, 1)
		b <- B{}

		var ok bool
		i, ok = <-b
		_ = ok

		i.M()
	}

	{
		i := I(A{})

		b := make(chan I, 1)
		b <- B{}

		select {
		case i = <-b:
		}

		i.M()
	}

	{
		i := I(A{})

		b := make(chan I, 1)
		b <- B{}

		var ok bool
		select {
		case i, ok = <-b:
		}
		_ = ok

		i.M()
	}
}

type I interface{ M() int }

type T int

func (T) M() int { return 0 }

type A struct{ T }
type B struct{ T }
