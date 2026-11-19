// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var trace []string

func src(s []int) []int { trace = append(trace, "src"); return s }

func size(n int) int { trace = append(trace, "size"); return n }

func main() {
	// The length is a call, inlined.
	_ = append(src([]int{9}), make([]int, size(2))...)
	check("call")

	// The length is a closure called in place, inlined by another path.
	_ = append(src([]int{9}), make([]int, func() int {
		trace = append(trace, "size")
		return 2
	}())...)
	check("closure")
}

func check(what string) {
	if len(trace) != 2 || trace[0] != "src" || trace[1] != "size" {
		println("after the", what, "append, trace has", len(trace), "entries, want 2:")
		for _, s := range trace {
			println("\t", s)
		}
		panic("append(x, make([]T, y)...) lost a side effect")
	}
	trace = nil
}
