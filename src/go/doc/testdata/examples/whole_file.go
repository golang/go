// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package foo_test

import "fmt"

type X int

func (X) Foo() {
}

func (X) TestBlah() {
}

func (X) BenchmarkFoo() {
}

func Example() {
	fmt.Println("Hello, world!")
	// Output: Hello, world!
}
