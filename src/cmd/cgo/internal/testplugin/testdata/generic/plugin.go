// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Instantiated function name may contain weird characters
// that confuse the external linker, so it needs to be
// mangled.

package main

//go:noinline
func F[T any]() {}

type S struct {
	X int `parser:"|@@)"`
}

func P() {
	F[S]()
}

func main() {}
