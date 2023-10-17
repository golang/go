// compile

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type myMap map[string] int;

func f() myMap {
	m := make(map[string] int);
	return m
}

func main() {
	m := make(myMap);
	mp := &m;

	{
		x, ok := m["key"];
		_, _ = x, ok;
	}
	{
		x, ok := (*mp)["key"];
		_, _ = x, ok;
	}
	{
		x, ok := f()["key"];
		_, _ = x, ok;
	}
	{
		var x int;
		var ok bool;
		x, ok = f()["key"];
		_, _ = x, ok;
	}
}

/*
 * bug143.go:19: assignment count mismatch: 2 = 1
 * bug143.go:18: x: undefined
 * bug143.go:18: ok: undefined
 */
