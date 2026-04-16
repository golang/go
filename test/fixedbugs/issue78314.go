// errorcheck

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func F() {
	for range (func(func(int, ...string) bool))(nil) { // ERROR "cannot be variadic"
	}
}

func main() {}
