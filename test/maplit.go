// run

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test map literals lookup

package main

import "fmt"

const (
	stringlit  = "a"
	intlit     = 22323
	float64lit = 34.3423
)

type S struct{ a int }

func main() {
	v1 := map[string]int{
		"a": 33333,
		"b": 66666,
	}[stringlit]
	if v1 != 33333 {
		panic(fmt.Sprintf("wanted %v got %v", 33333, v1))
	}

	v2 := map[int]string{
		4323:  "g",
		22323: "a",
	}[intlit]
	if v2 != "a" {
		panic(fmt.Sprintf("wanted %v got %v", "a", v2))
	}

	v3 := map[string]S{
		"f":  S{a: 3},
		"a":  S{a: 2231},
		"bb": S{a: 2},
	}[stringlit]
	if v3 != (S{a: 2231}) {
		panic(fmt.Sprintf("wanted %#v got %#v", S{a: 1}, v3))
	}

	v4 := map[float64]S{
		34.3423: S{a: 1},
		2.33:    S{a: 2},
	}[float64lit]
	if v4 != (S{a: 1}) {
		panic(fmt.Sprintf("wanted %#v got %#v", S{a: 1}, v4))
	}

	v5 := map[string]S{
		"f": S{a: 1},
		"v": S{a: 2},
	}["not_defined"] // Literal not defined on the map lit, should return type zero value.
	if v5 != (S{a: 0}) {
		panic(fmt.Sprintf("wanted %#v got %#v", S{a: 0}, v5))
	}
}
