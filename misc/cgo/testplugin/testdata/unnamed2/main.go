// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore

package main

// // No C code required.
import "C"

func FuncInt() int { return 2 }

func FuncRecursive() X { return X{} }

type Y struct {
	X *X
}
type X struct {
	Y Y
}

func main() {}
