// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "strings"

// an sexpr is an s-expression.  It is either a token or a
// parenthesized list of s-expressions.
//
// Used just for initial development.  Should we keep it for testing, or
// ditch it once we've plugged into the main compiler output?

type sexpr struct {
	compound bool
	name     string  // !compound
	parts    []sexpr // compound
}

func (s *sexpr) String() string {
	if !s.compound {
		return s.name
	}
	x := "("
	for i, p := range s.parts {
		if i != 0 {
			x += " "
		}
		x += p.String()
	}
	return x + ")"
}

func parseSexpr(s string) sexpr {
	var e string
	e, s = grabOne(s)
	if len(e) > 0 && e[0] == '(' {
		e = e[1 : len(e)-1]
		var parts []sexpr
		for e != "" {
			var p string
			p, e = grabOne(e)
			parts = append(parts, parseSexpr(p))
		}
		return sexpr{true, "", parts}
	}
	return sexpr{false, e, nil}
}

// grabOne peels off first token or parenthesized string from s.
// returns first thing and the remainder of s.
func grabOne(s string) (string, string) {
	for len(s) > 0 && s[0] == ' ' {
		s = s[1:]
	}
	if len(s) == 0 || s[0] != '(' {
		i := strings.Index(s, " ")
		if i < 0 {
			return s, ""
		}
		return s[:i], s[i:]
	}
	d := 0
	i := 0
	for {
		if len(s) == i {
			panic("unterminated s-expression: " + s)
		}
		if s[i] == '(' {
			d++
		}
		if s[i] == ')' {
			d--
			if d == 0 {
				i++
				return s[:i], s[i:]
			}
		}
		i++
	}
}
