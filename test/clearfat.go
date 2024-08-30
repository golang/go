// runoutput

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check that {5,6,8,9}g/ggen.c:clearfat is zeroing the entire object.

package main

import (
	"bytes"
	"fmt"
	"strconv"
	"strings"
)

const ntest = 1100

func main() {
	var decls, calls bytes.Buffer

	for i := 1; i <= ntest; i++ {
		s := strconv.Itoa(i)
		decls.WriteString(strings.Replace(decl, "$", s, -1))
		calls.WriteString(strings.Replace("poison$()\n\tclearfat$()\n\t", "$", s, -1))
	}

	program = strings.Replace(program, "$DECLS", decls.String(), 1)
	program = strings.Replace(program, "$CALLS", calls.String(), 1)
	fmt.Print(program)
}

var program = `package main

var count int

$DECLS

func main() {
	$CALLS
	if count != 0 {
		println("failed", count, "case(s)")
	}
}
`

const decl = `
func poison$() {
	// Grow and poison the stack space that will be used by clearfat$
	var t [2*$]byte
	for i := range t {
		t[i] = 0xff
	}
}

func clearfat$() {
	var t [$]byte

	for _, x := range t {
		if x != 0 {
//			println("clearfat$: index", i, "expected 0, got", x)
			count++
			break
		}
	}
}
`
