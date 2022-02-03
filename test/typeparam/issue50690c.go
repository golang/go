// run -gcflags=-G=3

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
)

type Printer[T ~string] struct {
	PrintFn func(T)
}

func Print[T ~string](s T) {
	fmt.Println(s)
}

func PrintWithPrinter[T ~string, S struct {
	ID      T
	PrintFn func(T)
}](message T, obj S) {
	obj.PrintFn(message)
}

func main() {
	PrintWithPrinter(
		"Hello, world.",
		struct {
			ID      string
			PrintFn func(string)
		}{ID: "fake", PrintFn: Print[string]},
	)
}
