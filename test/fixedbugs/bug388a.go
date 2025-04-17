// errorcheck

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 2231

package main
import "runtime"

func bar(i int) {
	runtime.UintType := i       // ERROR "non-name runtime.UintType|non-name on left side|undefined"
	println(runtime.UintType)	// ERROR "invalid use of type|undefined"
}

func baz() {
	main.i := 1	// ERROR "non-name main.i|non-name on left side|undefined"
	println(main.i)	// ERROR "no fields or methods|undefined"
}

func main() {
}
