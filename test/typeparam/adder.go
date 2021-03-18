// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
)

type AddType interface {
	type int, int64, string
}

// _Add can add numbers or strings
func _Add[T AddType](a, b T) T {
	return a + b
}

func main() {
	if got, want := _Add(5, 3), 8; got != want {
		panic(fmt.Sprintf("got %d, want %d", got, want))
	}
	if got, want := _Add("ab", "cd"), "abcd"; got != want {
		panic(fmt.Sprintf("got %d, want %d", got, want))
	}
}
