// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main


type T1[T any] struct {
	e innerT[T, T1[T]]
}

func main() {
	//Output:
	//recursive_constraints.go:4:6: invalid recursive type T1
	//recursive_constraints.go:4:6: T1 refers to
	//recursive_constraints.go:16:6: innerT refers to
	//recursive_constraints.go:4:6: T1
}

type innerT[T any, R T1[T]] struct {
	Ref R
}
