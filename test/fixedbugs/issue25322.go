// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Missing zero extension when converting a float32
// to a uint64.

package main

import (
	"fmt"
	"math"
)

func Foo(v float32) {
	fmt.Printf("%x\n", uint64(math.Float32bits(v)))
}

func main() {
	Foo(2.0)
}
