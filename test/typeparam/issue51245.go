// build

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T[P any] int
const C T[int] = 3

type T2 int
const C2 T2 = 9

func main() {
}
