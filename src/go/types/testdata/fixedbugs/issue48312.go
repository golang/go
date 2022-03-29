// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type T interface{ m() }
type P *T

func _(p *T) {
	p.m /* ERROR type \*T is pointer to interface, not interface */ ()
}

func _(p P) {
	p.m /* ERROR type P is pointer to interface, not interface */ ()
}

func _[P T](p *P) {
	p.m /* ERROR type \*P is pointer to type parameter, not type parameter */ ()
}
