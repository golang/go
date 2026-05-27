// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that a closure that captures a variable holding a constant still
// observes the right value once inlining made several copies of it, each
// capturing a different constant.

package main

import "fmt"

func str(s string) func() string {
	return func() string { return s }
}

func num(n int) func() int {
	return func() int { return n }
}

func iface(n int) func() any {
	return func() any { return n }
}

func nested(s string) func() func() string {
	return func() func() string {
		return func() string { return s }
	}
}

func main() {
	for i, f := range []func() string{str("a"), str("b"), str("c")} {
		if got, want := f(), string(rune('a'+i)); got != want {
			panic(fmt.Sprintf("str: got %q, want %q", got, want))
		}
	}

	for i, f := range []func() int{num(0), num(1), num(2)} {
		if got := f(); got != i {
			panic(fmt.Sprintf("num: got %d, want %d", got, i))
		}
	}

	for i, f := range []func() any{iface(0), iface(1), iface(2)} {
		if got := f(); got != any(i) {
			panic(fmt.Sprintf("iface: got %v, want %v", got, i))
		}
	}

	for i, f := range []func() func() string{nested("a"), nested("b"), nested("c")} {
		if got, want := f()(), string(rune('a'+i)); got != want {
			panic(fmt.Sprintf("nested: got %q, want %q", got, want))
		}
	}
}
