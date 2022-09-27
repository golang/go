// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

import (
	"fmt"
	"os"
)

type A struct {
	b int
}

func singleAssignment() {
	v := "s" // want `v declared (and|but) not used`

	s := []int{ // want `s declared (and|but) not used`
		1,
		2,
	}

	a := func(s string) bool { // want `a declared (and|but) not used`
		return false
	}

	if 1 == 1 {
		s := "v" // want `s declared (and|but) not used`
	}

	panic("I should survive")
}

func noOtherStmtsInBlock() {
	v := "s" // want `v declared (and|but) not used`
}

func partOfMultiAssignment() {
	f, err := os.Open("file") // want `f declared (and|but) not used`
	panic(err)
}

func sideEffects(cBool chan bool, cInt chan int) {
	b := <-c            // want `b declared (and|but) not used`
	s := fmt.Sprint("") // want `s declared (and|but) not used`
	a := A{             // want `a declared (and|but) not used`
		b: func() int {
			return 1
		}(),
	}
	c := A{<-cInt}          // want `c declared (and|but) not used`
	d := fInt() + <-cInt    // want `d declared (and|but) not used`
	e := fBool() && <-cBool // want `e declared (and|but) not used`
	f := map[int]int{       // want `f declared (and|but) not used`
		fInt(): <-cInt,
	}
	g := []int{<-cInt}       // want `g declared (and|but) not used`
	h := func(s string) {}   // want `h declared (and|but) not used`
	i := func(s string) {}() // want `i declared (and|but) not used`
}

func commentAbove() {
	// v is a variable
	v := "s" // want `v declared (and|but) not used`
}

func fBool() bool {
	return true
}

func fInt() int {
	return 1
}
