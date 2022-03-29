// run

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

func PrintWithPrinter[T ~string, S interface {
	~struct {
		ID       T
		PrintFn_ func(T)
	}
	PrintFn() func(T)
}](message T, obj S) {
	obj.PrintFn()(message)
}

func main() {
	PrintWithPrinter(
		"Hello, world.",
		StructWithPrinter{ID: "fake", PrintFn_: Print[string]},
	)
}

type StructWithPrinter struct {
	ID       string
	PrintFn_ func(string)
}

// Field accesses through type parameters are disabled
// until we have a more thorough understanding of the
// implications on the spec. See issue #51576.
// Use accessor method instead.

func (s StructWithPrinter) PrintFn() func(string) {
	return s.PrintFn_
}
