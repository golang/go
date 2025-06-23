// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// terminating statements

package stmt1

func _() {}

func _() int {} /* ERROR "missing return" */

func _() int { panic(0) }
func _() int { (panic(0)) }

// block statements
func _(x, y int) (z int) {
	{
		return
	}
}

func _(x, y int) (z int) {
	{
		return; ; ; // trailing empty statements are ok
	}
	; ; ;
}

func _(x, y int) (z int) {
	{
	}
} /* ERROR "missing return" */

func _(x, y int) (z int) {
	{
		; ; ;
	}
	; ; ;
} /* ERROR "missing return" */

// if statements
func _(x, y int) (z int) {
	if x < y { return }
	return 1
}

func _(x, y int) (z int) {
	if x < y { return; ; ; ; }
	return 1
}

func _(x, y int) (z int) {
	if x < y { return }
	return 1; ;
}

func _(x, y int) (z int) {
	if x < y { return }
} /* ERROR "missing return" */

func _(x, y int) (z int) {
	if x < y {
	} else { return 1
	}
} /* ERROR "missing return" */

func _(x, y int) (z int) {
	if x < y { return
	} else { return
	}
}

// for statements
func _(x, y int) (z int) {
	for x < y {
		return
	}
} /* ERROR "missing return" */

func _(x, y int) (z int) {
	for {
		return
	}
}

func _(x, y int) (z int) {
	for {
		return; ; ; ;
	}
}

func _(x, y int) (z int) {
	for {
		return
		break
	}
	; ; ;
} /* ERROR "missing return" */

func _(x, y int) (z int) {
	for {
		for { break }
		return
	}
}

func _(x, y int) (z int) {
	for {
		for { break }
		return ; ;
	}
	;
}

func _(x, y int) (z int) {
L:	for {
		for { break L }
		return
	}
} /* ERROR "missing return" */

// switch statements
func _(x, y int) (z int) {
	switch x {
	case 0: return
	default: return
	}
}

func _(x, y int) (z int) {
	switch x {
	case 0: return;
	default: return; ; ;
	}
}

func _(x, y int) (z int) {
	switch x {
	case 0: return
	}
} /* ERROR "missing return" */

func _(x, y int) (z int) {
	switch x {
	case 0: return
	case 1: break
	}
} /* ERROR "missing return" */

func _(x, y int) (z int) {
	switch x {
	case 0: return
	default:
		switch y {
		case 0: break
		}
		panic(0)
	}
}

func _(x, y int) (z int) {
	switch x {
	case 0: return
	default:
		switch y {
		case 0: break
		}
		panic(0); ; ;
	}
	;
}

func _(x, y int) (z int) {
L:	switch x {
	case 0: return
	default:
		switch y {
		case 0: break L
		}
		panic(0)
	}
} /* ERROR "missing return" */

// select statements
func _(ch chan int) (z int) {
	select {}
} // nice!

func _(ch chan int) (z int) {
	select {}
	; ;
}

func _(ch chan int) (z int) {
	select {
	default: break
	}
} /* ERROR "missing return" */

func _(ch chan int) (z int) {
	select {
	case <-ch: return
	default: break
	}
} /* ERROR "missing return" */

func _(ch chan int) (z int) {
	select {
	case <-ch: return
	default:
		for i := 0; i < 10; i++ {
			break
		}
		return
	}
}

func _(ch chan int) (z int) {
	select {
	case <-ch: return; ; ;
	default:
		for i := 0; i < 10; i++ {
			break
		}
		return; ; ;
	}
	; ; ;
}

func _(ch chan int) (z int) {
L:	select {
	case <-ch: return
	default:
		for i := 0; i < 10; i++ {
			break L
		}
		return
	}
	; ; ;
} /* ERROR "missing return" */

func parenPanic() int {
	((((((panic)))(0))))
}

func issue23218a() int {
	{
		panic := func(interface{}){}
		panic(0)
	}
} /* ERROR "missing return" */

func issue23218b() int {
	{
		panic := func(interface{}){}
		((((panic))))(0)
	}
} /* ERROR "missing return" */
