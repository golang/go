// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"os";
	"regexp";
)

var good_re = []string{
	``
,	`.`
,	`^.$`
,	`a`
,	`a*`
,	`a+`
,	`a?`
,	`a|b`
,	`a*|b*`
,	`(a*|b)(c*|d)`
,	`[a-z]`
,	`[a-abc-c\-\]\[]`
,	`[a-z]+`
,	`[]`
,	`[abc]`
,	`[^1234]`
}

// TODO: nice to do this with a map but we don't have an iterator
type StringError struct {
	re	string;
	err	*os.Error;
}
var bad_re = []StringError{
	StringError{ `*`,	 	regexp.ErrBareClosure },	
	StringError{ `(abc`,	regexp.ErrUnmatchedLpar },	
	StringError{ `abc)`,	regexp.ErrUnmatchedRpar },	
	StringError{ `x[a-z`,	regexp.ErrUnmatchedLbkt },	
	StringError{ `abc]`,	regexp.ErrUnmatchedRbkt },	
	StringError{ `[z-a]`,	regexp.ErrBadRange },	
	StringError{ `abc\`,	regexp.ErrExtraneousBackslash },	
	StringError{ `a**`,	regexp.ErrBadClosure },	
	StringError{ `a*+`,	regexp.ErrBadClosure },	
	StringError{ `a??`,	regexp.ErrBadClosure },	
	StringError{ `*`,	 	regexp.ErrBareClosure },	
	StringError{ `\x`,	regexp.ErrBadBackslash }
}

type Vec [20]int;

type Tester struct {
	re	string;
	text	string;
	match	Vec;
}

var matches = []Tester {
	Tester{ ``,	"",	Vec{0,0, -1,-1} },
	Tester{ `a`,	"a",	Vec{0,1, -1,-1} },
	Tester{ `b`,	"abc",	Vec{1,2, -1,-1} },
	Tester{ `.`,	"a",	Vec{0,1, -1,-1} },
	Tester{ `.*`,	"abcdef",	Vec{0,6, -1,-1} },
	Tester{ `^abcd$`,	"abcd",	Vec{0,4, -1,-1} },
	Tester{ `^bcd'`,	"abcdef",	Vec{-1,-1} },
	Tester{ `^abcd$`,	"abcde",	Vec{-1,-1} },
	Tester{ `a+`,	"baaab",	Vec{1, 4, -1,-1} },
	Tester{ `a*`,	"baaab",	Vec{0, 0, -1,-1} }
}

func Compile(expr string, error *os.Error) regexp.Regexp {
	re, err := regexp.Compile(expr);
	if err != error {
		print("compiling `", expr, "`; unexpected error: ", err.String(), "\n");
		sys.exit(1);
	}
	return re
}

func MarkedLen(m *[] int) int {
	if m == nil {
		return 0
	}
	var i int;
	for i = 0; i < len(m) && m[i] >= 0; i = i+2 {
	}
	return i
}

func PrintVec(m *[] int) {
	l := MarkedLen(m);
	for i := 0; i < l && m[i] >= 0; i = i+2 {
		print(m[i], ",", m[i+1], " ")
	}
}

func Equal(m1, m2 *[]int) bool {
	l := MarkedLen(m1);
	if l != MarkedLen(m2) {
		return false
	}
	for i := 0; i < l; i++ {
		if m1[i] != m2[i] {
			return false
		}
	}
	return true
}

func Match(expr string, str string, match *[]int) {
	re := Compile(expr, nil);
	m := re.Execute(str);
	if !Equal(m, match) {
		print("failure on `", expr, "` matching `", str, "`:\n");
		PrintVec(m);
		print("\nshould be:\n");
		PrintVec(match);
		print("\n");
		sys.exit(1);
	}
}

func main() {
	if sys.argc() > 1 {
		Compile(sys.argv(1), nil);
		sys.exit(0);
	}
	for i := 0; i < len(good_re); i++ {
		Compile(good_re[i], nil);
	}
	for i := 0; i < len(bad_re); i++ {
		Compile(bad_re[i].re, bad_re[i].err)
	}
	for i := 0; i < len(matches); i++ {
		t := &matches[i];
		Match(t.re, t.text, &t.match)
	}
}
