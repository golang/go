// run

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math/big"
)

//go:noinline
func f(x uint32) *big.Int {
	return big.NewInt(int64(x))
}
func main() {
	b := f(0xffffffff)
	c := big.NewInt(0xffffffff)
	if b.Cmp(c) != 0 {
		panic(fmt.Sprintf("b:%x c:%x", b, c))
	}
}
