// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"

	"./b"
)

// Test case for issue 30862.

// Be aware that unless GOEXPERIMENT=fieldtrack is set when building
// the compiler, this test will fail if executed with a regular GC
// compiler.

func main() {
	bad := b.Test()
	if len(bad) > 0 {
		for _, s := range bad {
			fmt.Fprintf(os.Stderr, "test failed: %s\n", s)
		}
		os.Exit(1)
	}
}
