// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// int the_answer();
import "C"
import (
	"fmt"
	"os"
)

func TheAnswer() int {
	return int(C.the_answer())
}

func main() {
	if a := TheAnswer(); a != 42 {
		fmt.Fprintln(os.Stderr, "Unexpected result for The Answer. Got:", a, " Want: 42")
		os.Exit(1)
	}
	fmt.Fprintln(os.Stdout, "ok")
}
