// run

// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"runtime"
	"strings"
)

type T struct {
	val int
}

func main() {
	defer expectError(22)
	var pT *T
	switch pT.val { // error should be here - line 22
	case 0:
		fmt.Println("0")
	case 1: // used to show up here instead
		fmt.Println("1")
	case 2:
		fmt.Println("2")
	}
	fmt.Println("finished")
}

func expectError(expectLine int) {
	if recover() == nil {
		panic("did not crash")
	}
	for i := 1;; i++ {
		_, file, line, ok := runtime.Caller(i)
		if !ok {
			panic("cannot find issue4562.go on stack")
		}
		if strings.HasSuffix(file, "issue4562.go") {
			if line != expectLine {
				panic(fmt.Sprintf("crashed at line %d, wanted line %d", line, expectLine))
			}
			break
		}
	}
}
