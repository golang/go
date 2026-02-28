// runoutput

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 10407: gccgo failed to remove carriage returns
// from raw string literals.

package main

import "fmt"

func main() {
	fmt.Println("package main\nfunc main() { if `a\rb\r\nc` != \"ab\\nc\" { panic(42) }}")
}
