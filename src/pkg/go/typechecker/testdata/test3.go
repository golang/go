// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package P3

// function and method signatures

func _()                                        {}
func _()                                        {}
func _(x, x /* ERROR "already declared" */ int) {}

func f()                                 {}
func f /* ERROR "already declared" */ () {}

func (*foo /* ERROR "invalid receiver" */ ) m() {}
func (bar /* ERROR "not a type" */ ) m()        {}

func f1(x, _, _ int) (_, _ float)                              {}
func f2(x, y, x /* ERROR "already declared" */ int)            {}
func f3(x, y int) (a, b, x /* ERROR "already declared" */ int) {}

func (x *T) m1()                                 {}
func (x *T) m1 /* ERROR "already declared" */ () {}
func (x T) m1 /* ERROR "already declared" */ ()  {}
func (T) m1 /* ERROR "already declared" */ ()    {}

func (x *T) m2(u, x /* ERROR "already declared" */ int)               {}
func (x *T) m3(a, b, c int) (u, x /* ERROR "already declared" */ int) {}
func (T) _(x, x /* ERROR "already declared" */ int)                   {}
func (T) _() (x, x /* ERROR "already declared" */ int)                {}

//func (PT) _() {}

var bar int

type T struct{}
type PT (T)
