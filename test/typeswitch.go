// $G $F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

const (
	Bool = iota;
	Int;
	Float;
	String;
	Struct;
	Chan;
	Array;
	Map;
	Func;
	Last;
)

type S struct { a int }
var s S = S{1234}

var c = make(chan int);

var a	= []int{0,1,2,3}

var m = make(map[string]int)

func assert(b bool, s string) {
	if !b {
		println(s);
		sys.Exit(1);
	}
}


func f(i int) interface{} {
	switch i {
	case Bool:
		return true;
	case Int:
		return 7;
	case Float:
		return 7.4;
	case String:
		return "hello";
	case Struct:
		return s;
	case Chan:
		return c;
	case Array:
		return a;
	case Map:
		return m;
	case Func:
		return f;
	}
	panic("bad type number");
}

func main() {
	// type guard style
//	for i := Bool; i < Last; i++ {
//		switch v := f(i); true {
//		case x := v.(bool):
//			assert(x == true && i == Bool, "switch 1 bool");
//		case x := v.(int):
//			assert(x == 7 && i == Int, "switch 1 int");
//		case x := v.(float):
//			assert(x == 7.4 && i == Float, "switch 1 float");
//		case x := v.(string):
//			assert(x == "hello" && i == String, "switch 1 string");
//		case x := v.(S):
//			assert(x.a == 1234 && i == Struct, "switch 1 struct");
//		case x := v.(chan int):
//			assert(x == c && i == Chan, "switch 1 chan");
//		case x := v.([]int):
//			assert(x[3] == 3 && i == Array, "switch 1 array");
//		case x := v.(map[string]int):
//			assert(x == m && i == Map, "switch 1 map");
//		case x := v.(func(i int) interface{}):
//			assert(x == f && i == Func, "switch 1 fun");
//		default:
//			assert(false, "switch 1 unknown");
//		}
//	}

	// type switch style
	for i := Bool; i < Last; i++ {
		switch x := f(i).(type) {
		case bool:
			assert(x == true && i == Bool, "switch 2 bool");
		case int:
			assert(x == 7 && i == Int, "switch 2 int");
		case float:
			assert(x == 7.4 && i == Float, "switch 2 float");
		case string:
			assert(x == "hello" && i == String, "switch 2 string");
		case S:
			assert(x.a == 1234 && i == Struct, "switch 2 struct");
		case chan int:
			assert(x == c && i == Chan, "switch 2 chan");
		case []int:
			assert(x[3] == 3 && i == Array, "switch 2 array");
		case map[string]int:
			assert(x == m && i == Map, "switch 2 map");
		case func(i int) interface{}:
			assert(x == f && i == Func, "switch 2 fun");
		default:
			assert(false, "switch 2 unknown");
		}
	}

	// catch-all style in various forms
	switch {
	case true:
		assert(true, "switch 3 bool");
	default:
		assert(false, "switch 3 unknown");
	}

	switch true {
	case true:
		assert(true, "switch 3 bool");
	default:
		assert(false, "switch 3 unknown");
	}

	switch false {
	case false:
		assert(true, "switch 4 bool");
	default:
		assert(false, "switch 4 unknown");
	}

//	switch true {
//	case x := f(Int).(float):
//		assert(false, "switch 5 type guard wrong type");
//	case x := f(Int).(int):
//		assert(x == 7, "switch 5 type guard");
//	default:
//		assert(false, "switch 5 unknown");
//	}

	m["7"] = 7;
//	switch true {
//	case x := m["6"]:
//		assert(false, "switch 6 map reference wrong");
//	case x := m["7"]:
//		assert(x == 7, "switch 6 map reference");
//	default:
//		assert(false, "switch 6 unknown");
//	}

	go func() { <-c; c <- 77; } ();
	// guarantee the channel is ready
	c <- 77;
	for i := 0; i < 5; i++ {
		sys.Gosched();
	}
	dummyc := make(chan int);
//	switch true {
//	case x := <-dummyc:
//		assert(false, "switch 7 chan wrong");
//	case x := <-c:
//		assert(x == 77, "switch 7 chan");
//	default:
//		assert(false, "switch 7 unknown");
//	}

}
