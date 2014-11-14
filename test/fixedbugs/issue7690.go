// run

// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 7690 - Stack and other routines did not back up initial PC
// into CALL instruction, instead reporting line number of next instruction,
// which might be on a different line.

package main

import (
	"bytes"
	"regexp"
	"runtime"
	"strconv"
)

func main() {
	buf1 := make([]byte, 1000)
	buf2 := make([]byte, 1000)

	runtime.Stack(buf1, false)      // CALL is last instruction on this line
	n := runtime.Stack(buf2, false) // CALL is followed by load of result from stack

	buf1 = buf1[:bytes.IndexByte(buf1, 0)]
	buf2 = buf2[:n]

	re := regexp.MustCompile(`(?m)^main\.main\(\)\n.*/issue7690.go:([0-9]+)`)
	m1 := re.FindStringSubmatch(string(buf1))
	if m1 == nil {
		println("BUG: cannot find main.main in first trace")
		return
	}
	m2 := re.FindStringSubmatch(string(buf2))
	if m2 == nil {
		println("BUG: cannot find main.main in second trace")
		return
	}

	n1, _ := strconv.Atoi(m1[1])
	n2, _ := strconv.Atoi(m2[1])
	if n1+1 != n2 {
		println("BUG: expect runtime.Stack on back to back lines, have", n1, n2)
		println(string(buf1))
		println(string(buf2))
	}
}
