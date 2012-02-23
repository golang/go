// errorcheck

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that pointers and interface types cannot be method receivers.
// Does not compile.

package main

type T struct {
	a int
}
type P *T
type P1 *T

func (p P) val() int   { return 1 } // ERROR "receiver.* pointer|invalid pointer or interface receiver"
func (p *P1) val() int { return 1 } // ERROR "receiver.* pointer|invalid pointer or interface receiver"

type I interface{}
type I1 interface{}

func (p I) val() int { return 1 } // ERROR "receiver.*interface|invalid pointer or interface receiver"
func (p *I1) val() int { return 1 } // ERROR "receiver.*interface|invalid pointer or interface receiver"

type Val interface {
	val() int
}

var _ = (*Val).val // ERROR "method"

var v Val
var pv = &v

var _ = pv.val()	// ERROR "method"
