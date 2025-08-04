// run

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func nan() float64 {
	var x, y float64
	return x / y
}

func main() {
	m := map[float64]int{}

	// Make a small map with nan keys
	for i := 0; i < 8; i++ {
		m[nan()] = i
	}

	// Start iterating on it.
	start := true
	for _, v := range m {
		if start {
			// Add some more elements.
			for i := 0; i < 10; i++ {
				m[float64(i)] = i
			}
			// Now clear the map.
			clear(m)
			start = false
		} else {
			// We should never reach here.
			panic(v)
		}
	}
}
