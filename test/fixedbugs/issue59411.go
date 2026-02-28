// run

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"reflect"
)

func main() {
	for i := 0; i < 100; i++ {
		f()
		g()
	}
}

func f() {
	// Allocate map.
	m := map[float64]int{}
	// Fill to just before a growth trigger.
	const N = 13 << 4 // 6.5 * 2 * 2^k
	for i := 0; i < N; i++ {
		m[math.NaN()] = i
	}
	// Trigger growth.
	m[math.NaN()] = N

	// Iterate through map.
	i := 0
	for range m {
		if i == 6 {
			// Partway through iteration, clear the map.
			clear(m)
		} else if i > 6 {
			// If we advance to the next iteration, that's a bug.
			panic("BAD")
		}
		i++
	}
	if len(m) != 0 {
		panic("clear did not empty the map")
	}
}

func g() {
	// Allocate map.
	m := map[float64]int{}
	// Fill to just before a growth trigger.
	const N = 13 << 4 // 6.5 * 2 * 2^k
	for i := 0; i < N; i++ {
		m[math.NaN()] = i
	}
	// Trigger growth.
	m[math.NaN()] = N

	// Iterate through map.
	i := 0
	v := reflect.ValueOf(m)
	iter := v.MapRange()
	for iter.Next() {
		if i == 6 {
			// Partway through iteration, clear the map.
			v.Clear()
		} else if i > 6 {
			// If we advance to the next iteration, that's a bug.
			panic("BAD")
		}
		i++
	}
	if v.Len() != 0 {
		panic("clear did not empty the map")
	}
}
