// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file is here just to cause problems.
// main.swig turns into a file also named main.go.
// Make sure cmd/go keeps them separate
// when both are passed to cgo.

package main

//int F(void) { return 1; }
import "C"
import (
	"fmt"
	"os"
)

func F() int { return int(C.F()) }

func main() {
	if x := int(C.F()); x != 1 {
		fatal("x = %d, want 1", x)
	}

	// Open this file itself and verify that the first few characters are
	// as expected.
	f := Fopen("main.go", "r")
	if f.Swigcptr() == 0 {
		fatal("fopen failed")
	}
	if Fgetc(f) != '/' || Fgetc(f) != '/' || Fgetc(f) != ' ' || Fgetc(f) != 'C' {
		fatal("read unexpected characters")
	}
	if Fclose(f) != 0 {
		fatal("fclose failed")
	}

	println("OK")
}

func fatal(f string, args ...any) {
	fmt.Fprintln(os.Stderr, fmt.Sprintf(f, args...))
	os.Exit(1)
}
