// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test simple type switches, including chans, maps etc.

package main

import "os"

const (
	Bool = iota
	Int
	Float
	String
	Struct
	Chan
	Array
	Map
	Func
	Last
)

type S struct {
	a int
}

var s S = S{1234}

var c = make(chan int)

var a = []int{0, 1, 2, 3}

var m = make(map[string]int)

func assert(b bool, s string) {
	if !b {
		println(s)
		os.Exit(1)
	}
}

func f(i int) interface{} {
	switch i {
	case Bool:
		return true
	case Int:
		return 7
	case Float:
		return 7.4
	case String:
		return "hello"
	case Struct:
		return s
	case Chan:
		return c
	case Array:
		return a
	case Map:
		return m
	case Func:
		return f
	}
	panic("bad type number")
}

func main() {
	for i := Bool; i < Last; i++ {
		switch x := f(i).(type) {
		case bool:
			assert(x == true && i == Bool, "bool")
		case int:
			assert(x == 7 && i == Int, "int")
		case float64:
			assert(x == 7.4 && i == Float, "float64")
		case string:
			assert(x == "hello" && i == String, "string")
		case S:
			assert(x.a == 1234 && i == Struct, "struct")
		case chan int:
			assert(x == c && i == Chan, "chan")
		case []int:
			assert(x[3] == 3 && i == Array, "array")
		case map[string]int:
			assert(x != nil && i == Map, "map")
		case func(i int) interface{}:
			assert(x != nil && i == Func, "fun")
		default:
			assert(false, "unknown")
		}
	}

	// boolean switch (has had bugs in past; worth writing down)
	switch {
	case true:
		assert(true, "switch 2 bool")
	default:
		assert(false, "switch 2 unknown")
	}

	switch true {
	case true:
		assert(true, "switch 3 bool")
	default:
		assert(false, "switch 3 unknown")
	}

	switch false {
	case false:
		assert(true, "switch 4 bool")
	default:
		assert(false, "switch 4 unknown")
	}
}
