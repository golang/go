// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type (
	A interface {
		a() interface {
			ABC1
		}
	}
	B interface {
		b() interface {
			ABC2
		}
	}
	C interface {
		c() interface {
			ABC3
		}
	}

	AB interface {
		A
		B
	}
	BC interface {
		B
		C
	}

	ABC1 interface {
		A
		B
		C
	}
	ABC2 interface {
		AB
		C
	}
	ABC3 interface {
		A
		BC
	}
)

var (
	x1 ABC1
	x2 ABC2
	x3 ABC3
)

func _() {
	// all types have the same method set
	x1 = x2
	x2 = x1

	x1 = x3
	x3 = x1

	x2 = x3
	x3 = x2

	// all methods return the same type again
	x1 = x1.a()
	x1 = x1.b()
	x1 = x1.c()

	x2 = x2.a()
	x2 = x2.b()
	x2 = x2.c()

	x3 = x3.a()
	x3 = x3.b()
	x3 = x3.c()
}
