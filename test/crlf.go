// $G $D/$F.go && $L $F.$A && ./$A.out >tmp.go &&
// $G tmp.go && $L tmp.$A && ./$A.out
// rm -f tmp.go

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test source files and strings containing \r and \r\n.

package main

import (
	"fmt"
	"strings"
)

func main() {
	prog = strings.Replace(prog, "BQ", "`", -1)
	prog = strings.Replace(prog, "CR", "\r", -1)
	fmt.Print(prog)
}

var prog = `
package main
CR

import "fmt"

var CR s = "hello\n" + CR
	" world"CR

var t = BQhelloCR
 worldBQ

var u = BQhCReCRlCRlCRoCR
 worldBQ

var golden = "hello\n world"

func main() {
	if s != golden {
		fmt.Printf("s=%q, want %q", s, golden)
	}
	if t != golden {
		fmt.Printf("t=%q, want %q", t, golden)
	}
	if u != golden {
		fmt.Printf("u=%q, want %q", u, golden)
	}
}
`
