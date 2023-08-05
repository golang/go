// errorcheck -0 -m=2

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func run() { // ERROR "cannot inline run: recursive"
	f := func() { // ERROR "can inline run.func1 with cost .* as:.*" "func literal does not escape"
		g() // ERROR "inlining call to g"
	}
	f() // ERROR "inlining call to run.func1" "inlining call to g"
	run()
}

func g() { // ERROR "can inline g with cost .* as:.*"
}

func main() { // ERROR "can inline main with cost .* as:.*"
	run()
}
