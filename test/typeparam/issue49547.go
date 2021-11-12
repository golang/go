// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

type foo int

func main() {
	want := "main.F[main.foo]"
	got := fmt.Sprintf("%T", F[foo]{})
	if got != want {
		fmt.Printf("want: %s, got: %s\n", want, got)
	}
}

type F[T any] struct {
}
