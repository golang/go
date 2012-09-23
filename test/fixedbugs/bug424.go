// $G $D/$F.dir/lib.go && $G $D/$F.go && $L $F.$A && ./$A.out

// NOTE: This test is not run by 'run.go' and so not run by all.bash.
// To run this test you must use the ./run shell script.

// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests that method calls through an interface always
// call the locally defined method localT.m independent
// at which embedding level it is and in which order
// embedding is done.

package main

import "./lib"
import "reflect"
import "fmt"

type localI interface {
	m() string
}

type localT struct{}

func (t *localT) m() string {
	return "main.localT.m"
}

type myT1 struct {
	localT
}

type myT2 struct {
	localT
	lib.T
}

type myT3 struct {
	lib.T
	localT
}

func main() {
	var i localI

	i = new(localT)
	if i.m() != "main.localT.m" {
		println("BUG: localT:", i.m(), "called")
	}

	i = new(myT1)
	if i.m() != "main.localT.m" {
		println("BUG: myT1:", i.m(), "called")
	}

	i = new(myT2)
	if i.m() != "main.localT.m" {
		println("BUG: myT2:", i.m(), "called")
	}

	t3 := new(myT3)
	if t3.m() != "main.localT.m" {
		println("BUG: t3:", t3.m(), "called")
	}
	
	i = new(myT3)
	if i.m() != "main.localT.m" {
		t := reflect.TypeOf(i)
		n := t.NumMethod()
		for j := 0; j < n; j++ {
			m := t.Method(j)
			fmt.Printf("#%d: %s.%s %s\n", j, m.PkgPath, m.Name, m.Type)
		}
		println("BUG: myT3:", i.m(), "called")
	}
	
	var t4 struct {
		localT
		lib.T
	}
	if t4.m() != "main.localT.m" {
		println("BUG: t4:", t4.m(), "called")
	}
	i = &t4
	if i.m() != "main.localT.m" {
		println("BUG: myT4:", i.m(), "called")
	}
	
	var t5 struct {
		lib.T
		localT
	}
	if t5.m() != "main.localT.m" {
		println("BUG: t5:", t5.m(), "called")
	}
	i = &t5
	if i.m() != "main.localT.m" {
		println("BUG: myT5:", i.m(), "called")
	}
}
