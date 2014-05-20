// run

// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
)

func main() {
	_, err := os.Stdout.Write(nil)
	if err != nil {
		fmt.Printf("BUG: os.Stdout.Write(nil) = %v\n", err)
	}
}
