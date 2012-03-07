// run

// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests that method calls through an interface always
// call the the locally defined method localT.m independent
// at which embedding level it is and in which order
// embedding is done.

package main

import "./bug424.dir"
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
}
