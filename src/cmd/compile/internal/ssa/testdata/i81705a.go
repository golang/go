// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func f(i int) bool { return i > 1 }

func g() {}

func main() {
	switch { // want a statement here, before the cases
	case f(1):
		g()
	case f(2):
		g()
	}
}
