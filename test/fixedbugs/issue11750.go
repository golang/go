// compile

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 11750: mkdotargslice: typecheck failed

package main

func main() {
	fn := func(names string) {

	}
	func(names ...string) {
		for _, name := range names {
			fn(name)
		}
	}("one", "two")
}
