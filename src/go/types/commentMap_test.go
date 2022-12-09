// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types_test

import (
	"fmt"
	"go/scanner"
	"go/token"
	"regexp"
	"strings"
	"testing"
)

type comment struct {
	line, col int    // comment position
	text      string // comment text, excluding "//", "/*", or "*/"
}

// commentMap collects all comments in the given src with comment text
// that matches the supplied regular expression rx and returns them as
// []comment lists in a map indexed by line number. The comment text is
// the comment with any comment markers ("//", "/*", or "*/") stripped.
// The position for each comment is the position of the token immediately
// preceding the comment, with all comments that are on the same line
// collected in a slice, in source order. If there is no preceding token
// (the matching comment appears at the beginning of the file), then the
// recorded position is unknown (line, col = 0, 0).
// If there are no matching comments, the result is nil.
func commentMap(src []byte, rx *regexp.Regexp) (res map[int][]comment) {
	fset := token.NewFileSet()
	file := fset.AddFile("", -1, len(src))

	var s scanner.Scanner
	s.Init(file, src, nil, scanner.ScanComments)
	var prev token.Pos // position of last non-comment, non-semicolon token

	for {
		pos, tok, lit := s.Scan()
		switch tok {
		case token.EOF:
			return
		case token.COMMENT:
			if lit[1] == '*' {
				lit = lit[:len(lit)-2] // strip trailing */
			}
			lit = lit[2:] // strip leading // or /*
			if rx.MatchString(lit) {
				p := fset.Position(prev)
				err := comment{p.Line, p.Column, lit}
				if res == nil {
					res = make(map[int][]comment)
				}
				res[p.Line] = append(res[p.Line], err)
			}
		case token.SEMICOLON:
			// ignore automatically inserted semicolon
			if lit == "\n" {
				continue
			}
			fallthrough
		default:
			prev = pos
		}
	}
}

func TestCommentMap(t *testing.T) {
	const src = `/* ERROR "0:0" */ /* ERROR "0:0" */ // ERROR "0:0"
// ERROR "0:0"
x /* ERROR "3:1" */                // ignore automatically inserted semicolon here
/* ERROR "3:1" */                  // position of x on previous line
   x /* ERROR "5:4" */ ;           // do not ignore this semicolon
/* ERROR "5:24" */                 // position of ; on previous line
	package /* ERROR "7:2" */  // indented with tab
        import  /* ERROR "8:9" */  // indented with blanks
`
	m := commentMap([]byte(src), regexp.MustCompile("^ ERROR "))
	found := 0 // number of errors found
	for line, errlist := range m {
		for _, err := range errlist {
			if err.line != line {
				t.Errorf("%v: got map line %d; want %d", err, err.line, line)
				continue
			}
			// err.line == line

			got := strings.TrimSpace(err.text[len(" ERROR "):])
			want := fmt.Sprintf(`"%d:%d"`, line, err.col)
			if got != want {
				t.Errorf("%v: got msg %q; want %q", err, got, want)
				continue
			}
			found++
		}
	}

	want := strings.Count(src, " ERROR ")
	if found != want {
		t.Errorf("commentMap got %d errors; want %d", found, want)
	}
}
