// errchk $G -e $D/$F.go

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type xyz struct {
    ch chan
} // ERROR "unexpected .*}.* in channel type"

func Foo(y chan) { // ERROR "unexpected .*\).* in channel type"
}

func Bar(x chan, y int) { // ERROR "unexpected comma in channel type"
}
