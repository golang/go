// $G $F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
	"unicode/utf8"
)

func main() {
	s := "\000\123\x00\xca\xFE\u0123\ubabe\U0000babe\U0010FFFFx"
	expect := []rune{0, 0123, 0, 0xFFFD, 0xFFFD, 0x123, 0xbabe, 0xbabe, 0x10FFFF, 'x'}
	offset := 0
	var i int
	var c rune
	ok := true
	cnum := 0
	for i, c = range s {
		r, size := utf8.DecodeRuneInString(s[i:len(s)]) // check it another way
		if i != offset {
			fmt.Printf("unexpected offset %d not %d\n", i, offset)
			ok = false
		}
		if r != expect[cnum] {
			fmt.Printf("unexpected rune %d from DecodeRuneInString: %x not %x\n", i, r, expect[cnum])
			ok = false
		}
		if c != expect[cnum] {
			fmt.Printf("unexpected rune %d from range: %x not %x\n", i, r, expect[cnum])
			ok = false
		}
		offset += size
		cnum++
	}
	if i != len(s)-1 {
		fmt.Println("after loop i is", i, "not", len(s)-1)
		ok = false
	}

	i = 12345
	c = 23456
	for i, c = range "" {
	}
	if i != 12345 {
		fmt.Println("range empty string assigned to index:", i)
		ok = false
	}
	if c != 23456 {
		fmt.Println("range empty string assigned to value:", c)
		ok = false
	}

	if !ok {
		fmt.Println("BUG: stringrange")
		os.Exit(1)
	}
}
