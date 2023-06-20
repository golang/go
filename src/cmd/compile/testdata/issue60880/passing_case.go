// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main


type innerT[T any, R *T1[T]] struct {
	Ref R
}

type T1[T any] struct {
	e innerT[T, *T1[T]]
}

func main() {
	//Output:
}
