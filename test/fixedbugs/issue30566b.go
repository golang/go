// run

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
)

func main() {
	_, _ = false || g(1), g(2)
	if !bytes.Equal(x, []byte{1, 2}) {
		panic(fmt.Sprintf("wanted [1,2], got %v", x))
	}
}

var x []byte

//go:noinline
func g(b byte) bool {
	x = append(x, b)
	return false
}
