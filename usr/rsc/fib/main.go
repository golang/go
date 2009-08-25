// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fib"

func main() {
	for i := 0; i < 10; i++ {
		println(fib.Fib(i));
	}
}
