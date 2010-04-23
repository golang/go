// [ $GOOS != nacl ] || exit 0  # NaCl cannot recover from signals
// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"runtime"
	"strings"
	"syscall"
)

var didbug bool

func bug() {
	if didbug {
		return
	}
	println("BUG")
	didbug = true
}

func check(name string, f func(), err string) {
	defer func() {
		v := recover()
		if v == nil {
			bug()
			println(name, "did not panic")
			return
		}
		runt, ok := v.(runtime.Error)
		if !ok {
			bug()
			println(name, "panicked but not with runtime.Error")
			return
		}
		s := runt.String()
		if strings.Index(s, err) < 0 {
			bug()
			println(name, "panicked with", s, "not", err)
			return
		}
	}()
	
	f()
}

func main() {
	var x int
	var x64 int64
	var p *[10]int
	var q *[10000]int
	var i int

	// not catching divide by zero on the arm.  is that even possible?
	if syscall.ARCH != "arm" {
		check("int-div-zero", func() { println(1/x) }, "integer divide by zero")
		check("int64-div-zero", func() { println(1/x64) }, "integer divide by zero")
	}

	check("nil-deref", func() { println(p[0]) }, "nil pointer dereference")
	check("nil-deref-1", func() { println(p[1]) }, "nil pointer dereference")
	check("nil-deref-big", func() { println(q[5000]) }, "nil pointer dereference")

	i = 99999
	var sl []int
	check("array-bounds", func() { println(p[i]) }, "index out of range")
	check("slice-bounds", func() { println(sl[i]) }, "index out of range")
	
	var inter interface{}
	inter = 1
	check("type-concrete", func() { println(inter.(string)) }, "int, not string")
	check("type-interface", func() { println(inter.(m)) }, "missing method m")
}

type m interface{ m() }
