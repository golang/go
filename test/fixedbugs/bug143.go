// $G $D/$F.go || echo BUG should compile

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type myMap map[string] int;

func f() *myMap {
	m := make(map[string] int);
	return &m
}

func main() {
	m := make(myMap);
	mp := &m;

	{
		x, ok := m["key"]
	}
	{
		x, ok := (*mp)["key"]
	}
	{
		x, ok := mp["key"]
	}
	{
		x, ok := f()["key"]
	}
	{
		var x int;
		var ok bool;
		x, ok = f()["key"]
	}
}

/*
 * bug143.go:19: assignment count mismatch: 2 = 1
 * bug143.go:18: x: undefined
 * bug143.go:18: ok: undefined
 */
