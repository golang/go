// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure we get the right line number for nil pointer panics.

package main

import (
	"fmt"
	"runtime/debug"
	"strings"
)

type T struct {
	a int
}

func (t *T) foo() {
	t.a = 3
}

func f1() {
	var t *T
	(*t).foo()
}

type U struct {
	V
}
type V struct {
	x int
}

func (v *V) setX() {
	v.x = 7
}

func f2() {
	var u *U
	u.setX()
}

func check(stack string, good []string, bad []string) {
	for _, g := range good {
		if !strings.Contains(stack, g) {
			fmt.Println(stack)
			fmt.Printf("ERROR: stack doesn't contain %s\n", g)
		}
	}
	for _, b := range bad {
		if strings.Contains(stack, b) {
			fmt.Println(stack)
			fmt.Printf("ERROR: stack contains %s\n", b)
		}
	}
}

func test1() {
	defer func() {
		r := recover()
		if r == nil {
			fmt.Println("ERROR: f1 should have panicked")
			return
		}
		check(string(debug.Stack()), []string{"f1", "issue79762.go:27"}, []string{"foo"})
	}()
	f1()
}

func test2() {
	defer func() {
		r := recover()
		if r == nil {
			fmt.Println("ERROR: f2 should have panicked")
			return
		}
		check(string(debug.Stack()), []string{"f2", "issue79762.go:43"}, []string{"setX"})
	}()
	f2()
}

func main() {
	test1()
	test2()
}
