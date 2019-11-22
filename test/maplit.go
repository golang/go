// run

// Copyright 2009 The Go Authors. All rights reserved.
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
		22323: "a",
		4323:  "g",
	}[intlit]
	if v2 != "a" {
		panic(fmt.Sprintf("wanted %v got %v", "a", v2))
	}

	v3 := map[float64]S{
		34.3423: S{a: 1},
		2.33:    S{a: 2},
	}[float64lit]
	if v3 != (S{a: 1}) {
		panic(fmt.Sprintf("wanted %#v got %#v", S{a: 1}, v3))
	}

	v4 := map[float64]S{
		34.3423: S{a: 1},
		2.33:    S{a: 2},
	}[34.34] // Literal not defined on the map lit
	if v4 != (S{a: 0}) {
		panic(fmt.Sprintf("wanted %#v got %#v", S{a: 0}, v4))
	}
}
