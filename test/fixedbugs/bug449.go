// runoutput

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 3866
// runtime.equal failed to take padding between arguments and
// return values into account, so in certain cases gc-generated
// code will read a random bool from the stack as the result of
// the comparison.
// This program generates a lot of equality tests and hopes to
// catch this.
// NOTE: this program assumes comparing instance of T and T's
// underlying []byte will make gc emit calls to runtime.equal,
// and if gc optimizes this case, then the test will no longer
// be correct (in the sense that it no longer tests runtime.equal).

package main

import (
	"bytes"
	"fmt"
	"strconv"
	"strings"
)

const ntest = 1024

func main() {
	var decls, calls bytes.Buffer

	for i := 1; i <= ntest; i++ {
		s := strconv.Itoa(i)
		decls.WriteString(strings.Replace(decl, "$", s, -1))
		calls.WriteString(strings.Replace("call(test$)\n\t", "$", s, -1))
	}

	program = strings.Replace(program, "$DECLS", decls.String(), 1)
	program = strings.Replace(program, "$CALLS", calls.String(), 1)
	fmt.Print(program)
}

var program = `package main

var count int

func call(f func() bool) {
	if f() {
		count++
	}
}

$DECLS

func main() {
	$CALLS
	if count != 0 {
		println("failed", count, "case(s)")
	}
}
`

const decl = `
type T$ [$]uint8
func test$() bool {
	v := T${1}
	return v == [$]uint8{2} || v != [$]uint8{1}
}`
