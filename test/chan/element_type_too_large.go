// errorcheck -e

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f() {
	n := len(make(chan struct{ v [65536]byte })) // ERROR "channel element type too large"
	_ = n
}

func g() {
	for v := range make(chan struct{ v [65536]byte }) { // ERROR "channel element type too large"
		_ = v
	}
}

func h(x int) {
	for v := 0; v < len(make(chan struct{ v [65536]byte })); v++ { // ERROR "channel element type too large"
	}
	for v := 0; v < 10; v += len(make(chan struct{ v [65536]byte })) { // ERROR "channel element type too large"
	}
	if v := 0; v < len(make(chan struct{ v [65536]byte })) { // ERROR "channel element type too large"
		_ = v
	}
	switch v := 0; v + len(make(chan struct{ v [65536]byte })) { // ERROR "channel element type too large"
	}
	switch v := 0; x {
	case v + len(make(chan struct{ v [65536]byte })): // ERROR "channel element type too large"
	}
	switch v := 0; y := any(make(chan struct{ v [65536]byte })).(type) { // ERROR "channel element type too large"
	default:
		_, _ = v, y
	}
	select {
	case v := <-make(chan struct{ v [65536]byte }): // ERROR "channel element type too large"
		_ = v
	case v, ok := <-make(chan struct{ v [65536]byte }): // ERROR "channel element type too large"
		_, _ = v, ok
	}
}

func multiline() {
	var a,
		b = 1, len(make(chan struct{ v [65536]byte })) // ERROR "channel element type too large"
	_, _ = a, b

	for a,
		b := 0, 0; a+b < len(make(chan struct{ v [65536]byte })); a++ { // ERROR "channel element type too large"
	}
	if a,
		b := 0, 0; a+b < len(make(chan struct{ v [65536]byte })) { // ERROR "channel element type too large"
		_, _ = a, b
	}
	switch a,
		b := 0, 0; a + b + len(make(chan struct{ v [65536]byte })) { // ERROR "channel element type too large"
	}
	switch v := 0;
	y := any(make(chan struct{ v [65536]byte })).(type) { // ERROR "channel element type too large"
	default:
		_, _ = v, y
	}
}
