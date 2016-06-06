// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package scanner_test

import (
	"fmt"
	"strings"
	"text/scanner"
)

func Example() {
	const src = `
	// This is scanned code.
	if a > 10 {
		someParsable = text
	}`
	var s scanner.Scanner
	s.Filename = "example"
	s.Init(strings.NewReader(src))
	var tok rune
	for tok != scanner.EOF {
		tok = s.Scan()
		fmt.Println("At position", s.Pos(), ":", s.TokenText())
	}

	// Output:
	// At position example:3:4 : if
	// At position example:3:6 : a
	// At position example:3:8 : >
	// At position example:3:11 : 10
	// At position example:3:13 : {
	// At position example:4:15 : someParsable
	// At position example:4:17 : =
	// At position example:4:22 : text
	// At position example:5:3 : }
	// At position example:5:3 :
}
