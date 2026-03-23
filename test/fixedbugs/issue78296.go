// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type A *C
type C = D[*H]
type D[_ any] struct{}
type H D[C]

func main() {}