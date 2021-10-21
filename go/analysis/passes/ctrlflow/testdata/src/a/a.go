// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

// This file tests facts produced by ctrlflow.

import (
	"log"
	"os"
	"runtime"
	"syscall"
	"testing"

	"lib"
)

var cond bool

func a() { // want a:"noReturn"
	if cond {
		b()
	} else {
		for {
		}
	}
}

func b() { // want b:"noReturn"
	select {}
}

func f(x int) { // no fact here
	switch x {
	case 0:
		os.Exit(0)
	case 1:
		panic(0)
	}
	// default case returns
}

type T int

func (T) method1() { // want method1:"noReturn"
	a()
}

func (T) method2() { // (may return)
	if cond {
		a()
	}
}

// Checking for the noreturn fact associated with F ensures that
// ctrlflow proved each of the listed functions was "noReturn".
//
func standardFunctions(x int) { // want standardFunctions:"noReturn"
	t := new(testing.T)
	switch x {
	case 0:
		t.FailNow()
	case 1:
		t.Fatal()
	case 2:
		t.Fatalf("")
	case 3:
		t.Skip()
	case 4:
		t.SkipNow()
	case 5:
		t.Skipf("")
	case 6:
		log.Fatal()
	case 7:
		log.Fatalf("")
	case 8:
		log.Fatalln()
	case 9:
		os.Exit(0)
	case 10:
		syscall.Exit(0)
	case 11:
		runtime.Goexit()
	case 12:
		log.Panic()
	case 13:
		log.Panicln()
	case 14:
		log.Panicf("")
	default:
		panic(0)
	}
}

// False positives are possible.
// This function is marked noReturn but in fact returns.
//
func spurious() { // want spurious:"noReturn"
	defer func() { recover() }()
	panic(nil)
}

func noBody()

func g() {
	lib.CanReturn()
}

func h() { // want h:"noReturn"
	lib.NoReturn()
}
