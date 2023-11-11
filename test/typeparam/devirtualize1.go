// run

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type S struct {
	x int
}

func (t *S) M1() {
}

func F[T any](x T) any {
	return x
}

func main() {
	F(&S{}).(interface{ M1() }).M1()
}
