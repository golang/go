// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

var (
	seen  []bool
	calls int
	sink  func()
)

//go:noinline
func wait(refusing bool) bool {
	seen = append(seen, refusing)
	return true
}

//go:noinline
func probe() bool {
	calls++
	return calls >= 3
}

func note() {
	if sink != nil {
		sink()
	}
}

func loop(first bool) {
	refusing := false
	for w := !first; ; w = true {
		if w && !wait(refusing) {
			note()
			return
		}
		if probe() {
			return
		}
		refusing = true
	}
}

func main() {
	for _, first := range []bool{false, true} {
		seen = nil
		calls = 0
		loop(first)
		want := "[false true true]"
		if first {
			want = "[true true]"
		}
		if got := fmt.Sprint(seen); got != want {
			panic(fmt.Sprintf("first=%v: got %s, want %s", first, got, want))
		}
	}
}
