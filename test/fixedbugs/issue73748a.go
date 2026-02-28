// run

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"context"
	"io"
	"runtime/trace"
)

type T struct {
	a [16]int
}

//go:noinline
func f(x *T) {
	*x = T{}
}

func main() {
	trace.Start(io.Discard)
	defer func() {
		recover()
		trace.Log(context.Background(), "a", "b")

	}()
	f(nil)
}
