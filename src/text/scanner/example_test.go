// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package scanner_test

import (
	"fmt"
	"strings"
	"text/scanner"
	"unicode"
)

func Example() {
	const src = `
// This is scanned code.
if a > 10 {
	someParsable = text
}`

	var s scanner.Scanner
	s.Init(strings.NewReader(src))
	s.Filename = "example"
	for tok := s.Scan(); tok != scanner.EOF; tok = s.Scan() {
		fmt.Printf("%s: %s\n", s.Position, s.TokenText())
	}

	// Output:
	// example:3:1: if
	// example:3:4: a
	// example:3:6: >
	// example:3:8: 10
	// example:3:11: {
	// example:4:2: someParsable
	// example:4:15: =
	// example:4:17: text
	// example:5:1: }
}

func Example_isIdentRune() {
	const src = "%var1 var2%"

	var s scanner.Scanner
	s.Init(strings.NewReader(src))
	s.Filename = "default"

	for tok := s.Scan(); tok != scanner.EOF; tok = s.Scan() {
		fmt.Printf("%s: %s\n", s.Position, s.TokenText())
	}

	fmt.Println()
	s.Init(strings.NewReader(src))
	s.Filename = "percent"

	// treat leading '%' as part of an identifier
	s.IsIdentRune = func(ch rune, i int) bool {
		return ch == '%' && i == 0 || unicode.IsLetter(ch) || unicode.IsDigit(ch) && i > 0
	}

	for tok := s.Scan(); tok != scanner.EOF; tok = s.Scan() {
		fmt.Printf("%s: %s\n", s.Position, s.TokenText())
	}

	// Output:
	// default:1:1: %
	// default:1:2: var1
	// default:1:7: var2
	// default:1:11: %
	//
	// percent:1:1: %var1
	// percent:1:7: var2
	// percent:1:11: %
}

func Example_mode() {
	const src = `
    // Comment begins at column 5.

This line should not be included in the output.

/*
This multiline comment
should be extracted in
its entirety.
*/
`

	var s scanner.Scanner
	s.Init(strings.NewReader(src))
	s.Filename = "comments"
	s.Mode ^= scanner.SkipComments // don't skip comments

	for tok := s.Scan(); tok != scanner.EOF; tok = s.Scan() {
		txt := s.TokenText()
		if strings.HasPrefix(txt, "//") || strings.HasPrefix(txt, "/*") {
			fmt.Printf("%s: %s\n", s.Position, txt)
		}
	}

	// Output:
	// comments:2:5: // Comment begins at column 5.
	// comments:6:1: /*
	// This multiline comment
	// should be extracted in
	// its entirety.
	// */
}

func Example_whitespace() {
	// tab-separated values
	const src = `aa	ab	ac	ad
ba	bb	bc	bd
ca	cb	cc	cd
da	db	dc	dd`

	var (
		col, row int
		s        scanner.Scanner
		tsv      [4][4]string // large enough for example above
	)
	s.Init(strings.NewReader(src))
	s.Whitespace ^= 1<<'\t' | 1<<'\n' // don't skip tabs and new lines

	for tok := s.Scan(); tok != scanner.EOF; tok = s.Scan() {
		switch tok {
		case '\n':
			row++
			col = 0
		case '\t':
			col++
		default:
			tsv[row][col] = s.TokenText()
		}
	}

	fmt.Print(tsv)

	// Output:
	// [[aa ab ac ad] [ba bb bc bd] [ca cb cc cd] [da db dc dd]]
}
