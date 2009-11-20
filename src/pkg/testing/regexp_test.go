// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing

import (
	"strings";
)

var good_re = []string{
	``,
	`.`,
	`^.$`,
	`a`,
	`a*`,
	`a+`,
	`a?`,
	`a|b`,
	`a*|b*`,
	`(a*|b)(c*|d)`,
	`[a-z]`,
	`[a-abc-c\-\]\[]`,
	`[a-z]+`,
	`[]`,
	`[abc]`,
	`[^1234]`,
	`[^\n]`,
}

// TODO: nice to do this with a map
type stringError struct {
	re	string;
	err	string;
}

var bad_re = []stringError{
	stringError{`*`, ErrBareClosure},
	stringError{`(abc`, ErrUnmatchedLpar},
	stringError{`abc)`, ErrUnmatchedRpar},
	stringError{`x[a-z`, ErrUnmatchedLbkt},
	stringError{`abc]`, ErrUnmatchedRbkt},
	stringError{`[z-a]`, ErrBadRange},
	stringError{`abc\`, ErrExtraneousBackslash},
	stringError{`a**`, ErrBadClosure},
	stringError{`a*+`, ErrBadClosure},
	stringError{`a??`, ErrBadClosure},
	stringError{`*`, ErrBareClosure},
	stringError{`\x`, ErrBadBackslash},
}

type vec []int

type tester struct {
	re	string;
	text	string;
	match	vec;
}

var matches = []tester{
	tester{``, "", vec{0, 0}},
	tester{`a`, "a", vec{0, 1}},
	tester{`x`, "y", vec{}},
	tester{`b`, "abc", vec{1, 2}},
	tester{`.`, "a", vec{0, 1}},
	tester{`.*`, "abcdef", vec{0, 6}},
	tester{`^abcd$`, "abcd", vec{0, 4}},
	tester{`^bcd'`, "abcdef", vec{}},
	tester{`^abcd$`, "abcde", vec{}},
	tester{`a+`, "baaab", vec{1, 4}},
	tester{`a*`, "baaab", vec{0, 0}},
	tester{`[a-z]+`, "abcd", vec{0, 4}},
	tester{`[^a-z]+`, "ab1234cd", vec{2, 6}},
	tester{`[a\-\]z]+`, "az]-bcz", vec{0, 4}},
	tester{`[^\n]+`, "abcd\n", vec{0, 4}},
	tester{`[日本語]+`, "日本語日本語", vec{0, 18}},
	tester{`()`, "", vec{0, 0, 0, 0}},
	tester{`(a)`, "a", vec{0, 1, 0, 1}},
	tester{`(.)(.)`, "日a", vec{0, 4, 0, 3, 3, 4}},
	tester{`(.*)`, "", vec{0, 0, 0, 0}},
	tester{`(.*)`, "abcd", vec{0, 4, 0, 4}},
	tester{`(..)(..)`, "abcd", vec{0, 4, 0, 2, 2, 4}},
	tester{`(([^xyz]*)(d))`, "abcd", vec{0, 4, 0, 4, 0, 3, 3, 4}},
	tester{`((a|b|c)*(d))`, "abcd", vec{0, 4, 0, 4, 2, 3, 3, 4}},
	tester{`(((a|b|c)*)(d))`, "abcd", vec{0, 4, 0, 4, 0, 3, 2, 3, 3, 4}},
	tester{`a*(|(b))c*`, "aacc", vec{0, 4, 2, 2, -1, -1}},
}

func compileTest(t *T, expr string, error string) *Regexp {
	re, err := CompileRegexp(expr);
	if err != error {
		t.Error("compiling `", expr, "`; unexpected error: ", err)
	}
	return re;
}

func printVec(t *T, m []int) {
	l := len(m);
	if l == 0 {
		t.Log("\t<no match>")
	} else {
		for i := 0; i < l; i = i + 2 {
			t.Log("\t", m[i], ",", m[i+1])
		}
	}
}

func printStrings(t *T, m []string) {
	l := len(m);
	if l == 0 {
		t.Log("\t<no match>")
	} else {
		for i := 0; i < l; i = i + 2 {
			t.Logf("\t%q", m[i])
		}
	}
}

func printBytes(t *T, b [][]byte) {
	l := len(b);
	if l == 0 {
		t.Log("\t<no match>")
	} else {
		for i := 0; i < l; i = i + 2 {
			t.Logf("\t%q", b[i])
		}
	}
}

func equal(m1, m2 []int) bool {
	l := len(m1);
	if l != len(m2) {
		return false
	}
	for i := 0; i < l; i++ {
		if m1[i] != m2[i] {
			return false
		}
	}
	return true;
}

func equalStrings(m1, m2 []string) bool {
	l := len(m1);
	if l != len(m2) {
		return false
	}
	for i := 0; i < l; i++ {
		if m1[i] != m2[i] {
			return false
		}
	}
	return true;
}

func equalBytes(m1 [][]byte, m2 []string) bool {
	l := len(m1);
	if l != len(m2) {
		return false
	}
	for i := 0; i < l; i++ {
		if string(m1[i]) != m2[i] {
			return false
		}
	}
	return true;
}

func executeTest(t *T, expr string, str string, match []int) {
	re := compileTest(t, expr, "");
	if re == nil {
		return
	}
	m := re.ExecuteString(str);
	if !equal(m, match) {
		t.Error("ExecuteString failure on `", expr, "` matching `", str, "`:");
		printVec(t, m);
		t.Log("should be:");
		printVec(t, match);
	}
	// now try bytes
	m = re.Execute(strings.Bytes(str));
	if !equal(m, match) {
		t.Error("Execute failure on `", expr, "` matching `", str, "`:");
		printVec(t, m);
		t.Log("should be:");
		printVec(t, match);
	}
}

func TestGoodCompile(t *T) {
	for i := 0; i < len(good_re); i++ {
		compileTest(t, good_re[i], "")
	}
}

func TestBadCompile(t *T) {
	for i := 0; i < len(bad_re); i++ {
		compileTest(t, bad_re[i].re, bad_re[i].err)
	}
}

func TestExecute(t *T) {
	for i := 0; i < len(matches); i++ {
		test := &matches[i];
		executeTest(t, test.re, test.text, test.match);
	}
}

func matchTest(t *T, expr string, str string, match []int) {
	re := compileTest(t, expr, "");
	if re == nil {
		return
	}
	m := re.MatchString(str);
	if m != (len(match) > 0) {
		t.Error("MatchString failure on `", expr, "` matching `", str, "`:", m, "should be", len(match) > 0)
	}
	// now try bytes
	m = re.Match(strings.Bytes(str));
	if m != (len(match) > 0) {
		t.Error("Match failure on `", expr, "` matching `", str, "`:", m, "should be", len(match) > 0)
	}
}

func TestMatch(t *T) {
	for i := 0; i < len(matches); i++ {
		test := &matches[i];
		matchTest(t, test.re, test.text, test.match);
	}
}

func matchStringsTest(t *T, expr string, str string, match []int) {
	re := compileTest(t, expr, "");
	if re == nil {
		return
	}
	strs := make([]string, len(match)/2);
	for i := 0; i < len(match); i++ {
		strs[i/2] = str[match[i]:match[i+1]]
	}
	m := re.MatchStrings(str);
	if !equalStrings(m, strs) {
		t.Error("MatchStrings failure on `", expr, "` matching `", str, "`:");
		printStrings(t, m);
		t.Log("should be:");
		printStrings(t, strs);
	}
	// now try bytes
	s := re.MatchSlices(strings.Bytes(str));
	if !equalBytes(s, strs) {
		t.Error("MatchSlices failure on `", expr, "` matching `", str, "`:");
		printBytes(t, s);
		t.Log("should be:");
		printStrings(t, strs);
	}
}

func TestMatchStrings(t *T) {
	for i := 0; i < len(matches); i++ {
		test := &matches[i];
		matchTest(t, test.re, test.text, test.match);
	}
}

func matchFunctionTest(t *T, expr string, str string, match []int) {
	m, err := MatchString(expr, str);
	if err == "" {
		return
	}
	if m != (len(match) > 0) {
		t.Error("function Match failure on `", expr, "` matching `", str, "`:", m, "should be", len(match) > 0)
	}
}

func TestMatchFunction(t *T) {
	for i := 0; i < len(matches); i++ {
		test := &matches[i];
		matchFunctionTest(t, test.re, test.text, test.match);
	}
}

func BenchmarkSimpleMatch(b *B) {
	b.StopTimer();
	re, _ := CompileRegexp("a");
	b.StartTimer();

	for i := 0; i < b.N; i++ {
		re.MatchString("a")
	}
}

func BenchmarkUngroupedMatch(b *B) {
	b.StopTimer();
	re, _ := CompileRegexp("[a-z]+ [0-9]+ [a-z]+");
	b.StartTimer();

	for i := 0; i < b.N; i++ {
		re.MatchString("word 123 other")
	}
}

func BenchmarkGroupedMatch(b *B) {
	b.StopTimer();
	re, _ := CompileRegexp("([a-z]+) ([0-9]+) ([a-z]+)");
	b.StartTimer();

	for i := 0; i < b.N; i++ {
		re.MatchString("word 123 other")
	}
}
