// errchk $G $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Error messages about missing implicit methods.

package main

type T int

func (t T) V()
func (t *T) P()

type V interface {
	V()
}
type P interface {
	P()
	V()
}

type S struct {
	T
}
type SP struct {
	*T
}

func main() {
	var t T
	var v V
	var p P
	var s S
	var sp SP

	v = t
	p = t // ERROR "does not implement|requires a pointer"
	_, _ = v, p
	v = &t
	p = &t
	_, _ = v, p

	v = s
	p = s // ERROR "does not implement|requires a pointer"
	_, _ = v, p
	v = &s
	p = &s
	_, _ = v, p

	v = sp
	p = sp // no error!
	_, _ = v, p
	v = &sp
	p = &sp
	_, _ = v, p
}
