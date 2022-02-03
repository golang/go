// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"

	"a1"
	"a2"
)

func New() int {
	return a1.New() + a2.New()
}

func main() {
	if got, want := New(), 0; got != want {
		panic(fmt.Sprintf("got %d, want %d", got, want))
	}
}
