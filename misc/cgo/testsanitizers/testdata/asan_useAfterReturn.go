// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// The -fsanitize=address option of C compier can detect stack-use-after-return bugs.
// In the following program, the local variable 'local' was moved to heap by the Go
// compiler because foo() is returning the reference to 'local', and return stack of
// foo() will be invalid. Thus for main() to use the reference to 'local', the 'local'
// must be available even after foo() has finished. Therefore, Go has no such issue.

import "fmt"

var ptr *int

func main() {
	foo()
	fmt.Printf("ptr=%x, %v", *ptr, ptr)
}

func foo() {
	var local int
	local = 1
	ptr = &local // local is moved to heap.
}
