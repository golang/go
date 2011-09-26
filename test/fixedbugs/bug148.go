// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T struct {a, b int};

func println(x, y int) { }

func f(x interface{}) interface{} {
	type T struct {a, b int};

	if x == nil {
		return T{2, 3};
	}

	t := x.(T);
	println(t.a, t.b);
	return x;
}

func main() {
	inner_T := f(nil);
	f(inner_T);

	shouldPanic(p1)
}

func p1() {
	outer_T := T{5, 7};
	f(outer_T);
}

func shouldPanic(f func()) {
	defer func() {
		if recover() == nil {
			panic("function should panic")
		}
	}()
	f()
}

/*
This prints:

2 3
5 7

but it should crash: The type assertion on line 18 should fail
for the 2nd call to f with outer_T.
*/
