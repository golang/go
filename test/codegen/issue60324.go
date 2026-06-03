// asmcheck

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

func main() {
	// amd64:"LEAQ command-line-arguments\\.h\\.func1"
	f(1)()

	// amd64:"LEAQ command-line-arguments\\.h\\.func1"
	g(2)()

	// amd64:"LEAQ command-line-arguments\\.h\\.func1"
	h(3)()

	// amd64:"LEAQ command-line-arguments\\.h\\.func1"
	f(4)()
}

func f(x int) func() {
	// amd64:"LEAQ command-line-arguments\\.h\\.func1"
	return g(x)
}

func g(x int) func() {
	// amd64:"LEAQ command-line-arguments\\.h\\.func1"
	return h(x)
}

func h(x int) func() {
	// amd64:"LEAQ command-line-arguments\\.h\\.func1"
	return func() { recover() }
}
