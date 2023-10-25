// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "unsafe"

var (
	_ A = A(nil).a().b().c().d().e().f()
	_ A = A(nil).b().c().d().e().f()
	_ A = A(nil).c().d().e().f()
	_ A = A(nil).d().e().f()
	_ A = A(nil).e().f()
	_ A = A(nil).f()
	_ A = A(nil)
)

type (
	A interface {
		a() B
		B
	}

	B interface {
		b() C
		C
	}

	C interface {
		c() D
		D
	}

	D interface {
		d() E
		E
	}

	E interface {
		e() F
		F
	}

	F interface {
		f() A
	}
)

type (
	U /* ERROR "invalid recursive type" */ interface {
		V
	}

	V interface {
		v() [unsafe.Sizeof(u)]int
	}
)

var u U
