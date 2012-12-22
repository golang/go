// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regexp

import (
	"reflect"
	"strings"
	"testing"
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
	`[abc]`,
	`[^1234]`,
	`[^\n]`,
	`\!\\`,
}

type stringError struct {
	re  string
	err string
}

var bad_re = []stringError{
	{`*`, "missing argument to repetition operator: `*`"},
	{`+`, "missing argument to repetition operator: `+`"},
	{`?`, "missing argument to repetition operator: `?`"},
	{`(abc`, "missing closing ): `(abc`"},
	{`abc)`, "unexpected ): `abc)`"},
	{`x[a-z`, "missing closing ]: `[a-z`"},
	{`[z-a]`, "invalid character class range: `z-a`"},
	{`abc\`, "trailing backslash at end of expression"},
	{`a**`, "invalid nested repetition operator: `**`"},
	{`a*+`, "invalid nested repetition operator: `*+`"},
	{`\x`, "invalid escape sequence: `\\x`"},
}

func compileTest(t *testing.T, expr string, error string) *Regexp {
	re, err := Compile(expr)
	if error == "" && err != nil {
		t.Error("compiling `", expr, "`; unexpected error: ", err.Error())
	}
	if error != "" && err == nil {
		t.Error("compiling `", expr, "`; missing error")
	} else if error != "" && !strings.Contains(err.Error(), error) {
		t.Error("compiling `", expr, "`; wrong error: ", err.Error(), "; want ", error)
	}
	return re
}

func TestGoodCompile(t *testing.T) {
	for i := 0; i < len(good_re); i++ {
		compileTest(t, good_re[i], "")
	}
}

func TestBadCompile(t *testing.T) {
	for i := 0; i < len(bad_re); i++ {
		compileTest(t, bad_re[i].re, bad_re[i].err)
	}
}

func matchTest(t *testing.T, test *FindTest) {
	re := compileTest(t, test.pat, "")
	if re == nil {
		return
	}
	m := re.MatchString(test.text)
	if m != (len(test.matches) > 0) {
		t.Errorf("MatchString failure on %s: %t should be %t", test, m, len(test.matches) > 0)
	}
	// now try bytes
	m = re.Match([]byte(test.text))
	if m != (len(test.matches) > 0) {
		t.Errorf("Match failure on %s: %t should be %t", test, m, len(test.matches) > 0)
	}
}

func TestMatch(t *testing.T) {
	for _, test := range findTests {
		matchTest(t, &test)
	}
}

func matchFunctionTest(t *testing.T, test *FindTest) {
	m, err := MatchString(test.pat, test.text)
	if err == nil {
		return
	}
	if m != (len(test.matches) > 0) {
		t.Errorf("Match failure on %s: %t should be %t", test, m, len(test.matches) > 0)
	}
}

func TestMatchFunction(t *testing.T) {
	for _, test := range findTests {
		matchFunctionTest(t, &test)
	}
}

type ReplaceTest struct {
	pattern, replacement, input, output string
}

var replaceTests = []ReplaceTest{
	// Test empty input and/or replacement, with pattern that matches the empty string.
	{"", "", "", ""},
	{"", "x", "", "x"},
	{"", "", "abc", "abc"},
	{"", "x", "abc", "xaxbxcx"},

	// Test empty input and/or replacement, with pattern that does not match the empty string.
	{"b", "", "", ""},
	{"b", "x", "", ""},
	{"b", "", "abc", "ac"},
	{"b", "x", "abc", "axc"},
	{"y", "", "", ""},
	{"y", "x", "", ""},
	{"y", "", "abc", "abc"},
	{"y", "x", "abc", "abc"},

	// Multibyte characters -- verify that we don't try to match in the middle
	// of a character.
	{"[a-c]*", "x", "\u65e5", "x\u65e5x"},
	{"[^\u65e5]", "x", "abc\u65e5def", "xxx\u65e5xxx"},

	// Start and end of a string.
	{"^[a-c]*", "x", "abcdabc", "xdabc"},
	{"[a-c]*$", "x", "abcdabc", "abcdx"},
	{"^[a-c]*$", "x", "abcdabc", "abcdabc"},
	{"^[a-c]*", "x", "abc", "x"},
	{"[a-c]*$", "x", "abc", "x"},
	{"^[a-c]*$", "x", "abc", "x"},
	{"^[a-c]*", "x", "dabce", "xdabce"},
	{"[a-c]*$", "x", "dabce", "dabcex"},
	{"^[a-c]*$", "x", "dabce", "dabce"},
	{"^[a-c]*", "x", "", "x"},
	{"[a-c]*$", "x", "", "x"},
	{"^[a-c]*$", "x", "", "x"},

	{"^[a-c]+", "x", "abcdabc", "xdabc"},
	{"[a-c]+$", "x", "abcdabc", "abcdx"},
	{"^[a-c]+$", "x", "abcdabc", "abcdabc"},
	{"^[a-c]+", "x", "abc", "x"},
	{"[a-c]+$", "x", "abc", "x"},
	{"^[a-c]+$", "x", "abc", "x"},
	{"^[a-c]+", "x", "dabce", "dabce"},
	{"[a-c]+$", "x", "dabce", "dabce"},
	{"^[a-c]+$", "x", "dabce", "dabce"},
	{"^[a-c]+", "x", "", ""},
	{"[a-c]+$", "x", "", ""},
	{"^[a-c]+$", "x", "", ""},

	// Other cases.
	{"abc", "def", "abcdefg", "defdefg"},
	{"bc", "BC", "abcbcdcdedef", "aBCBCdcdedef"},
	{"abc", "", "abcdabc", "d"},
	{"x", "xXx", "xxxXxxx", "xXxxXxxXxXxXxxXxxXx"},
	{"abc", "d", "", ""},
	{"abc", "d", "abc", "d"},
	{".+", "x", "abc", "x"},
	{"[a-c]*", "x", "def", "xdxexfx"},
	{"[a-c]+", "x", "abcbcdcdedef", "xdxdedef"},
	{"[a-c]*", "x", "abcbcdcdedef", "xdxdxexdxexfx"},

	// Substitutions
	{"a+", "($0)", "banana", "b(a)n(a)n(a)"},
	{"a+", "(${0})", "banana", "b(a)n(a)n(a)"},
	{"a+", "(${0})$0", "banana", "b(a)an(a)an(a)a"},
	{"a+", "(${0})$0", "banana", "b(a)an(a)an(a)a"},
	{"hello, (.+)", "goodbye, ${1}", "hello, world", "goodbye, world"},
	{"hello, (.+)", "goodbye, $1x", "hello, world", "goodbye, "},
	{"hello, (.+)", "goodbye, ${1}x", "hello, world", "goodbye, worldx"},
	{"hello, (.+)", "<$0><$1><$2><$3>", "hello, world", "<hello, world><world><><>"},
	{"hello, (?P<noun>.+)", "goodbye, $noun!", "hello, world", "goodbye, world!"},
	{"hello, (?P<noun>.+)", "goodbye, ${noun}", "hello, world", "goodbye, world"},
	{"(?P<x>hi)|(?P<x>bye)", "$x$x$x", "hi", "hihihi"},
	{"(?P<x>hi)|(?P<x>bye)", "$x$x$x", "bye", "byebyebye"},
	{"(?P<x>hi)|(?P<x>bye)", "$xyz", "hi", ""},
	{"(?P<x>hi)|(?P<x>bye)", "${x}yz", "hi", "hiyz"},
	{"(?P<x>hi)|(?P<x>bye)", "hello $$x", "hi", "hello $x"},
	{"a+", "${oops", "aaa", "${oops"},
	{"a+", "$$", "aaa", "$"},
	{"a+", "$", "aaa", "$"},

	// Substitution when subexpression isn't found
	{"(x)?", "$1", "123", "123"},
	{"abc", "$1", "123", "123"},
}

var replaceLiteralTests = []ReplaceTest{
	// Substitutions
	{"a+", "($0)", "banana", "b($0)n($0)n($0)"},
	{"a+", "(${0})", "banana", "b(${0})n(${0})n(${0})"},
	{"a+", "(${0})$0", "banana", "b(${0})$0n(${0})$0n(${0})$0"},
	{"a+", "(${0})$0", "banana", "b(${0})$0n(${0})$0n(${0})$0"},
	{"hello, (.+)", "goodbye, ${1}", "hello, world", "goodbye, ${1}"},
	{"hello, (?P<noun>.+)", "goodbye, $noun!", "hello, world", "goodbye, $noun!"},
	{"hello, (?P<noun>.+)", "goodbye, ${noun}", "hello, world", "goodbye, ${noun}"},
	{"(?P<x>hi)|(?P<x>bye)", "$x$x$x", "hi", "$x$x$x"},
	{"(?P<x>hi)|(?P<x>bye)", "$x$x$x", "bye", "$x$x$x"},
	{"(?P<x>hi)|(?P<x>bye)", "$xyz", "hi", "$xyz"},
	{"(?P<x>hi)|(?P<x>bye)", "${x}yz", "hi", "${x}yz"},
	{"(?P<x>hi)|(?P<x>bye)", "hello $$x", "hi", "hello $$x"},
	{"a+", "${oops", "aaa", "${oops"},
	{"a+", "$$", "aaa", "$$"},
	{"a+", "$", "aaa", "$"},
}

type ReplaceFuncTest struct {
	pattern       string
	replacement   func(string) string
	input, output string
}

var replaceFuncTests = []ReplaceFuncTest{
	{"[a-c]", func(s string) string { return "x" + s + "y" }, "defabcdef", "defxayxbyxcydef"},
	{"[a-c]+", func(s string) string { return "x" + s + "y" }, "defabcdef", "defxabcydef"},
	{"[a-c]*", func(s string) string { return "x" + s + "y" }, "defabcdef", "xydxyexyfxabcydxyexyfxy"},
}

func TestReplaceAll(t *testing.T) {
	for _, tc := range replaceTests {
		re, err := Compile(tc.pattern)
		if err != nil {
			t.Errorf("Unexpected error compiling %q: %v", tc.pattern, err)
			continue
		}
		actual := re.ReplaceAllString(tc.input, tc.replacement)
		if actual != tc.output {
			t.Errorf("%q.ReplaceAllString(%q,%q) = %q; want %q",
				tc.pattern, tc.input, tc.replacement, actual, tc.output)
		}
		// now try bytes
		actual = string(re.ReplaceAll([]byte(tc.input), []byte(tc.replacement)))
		if actual != tc.output {
			t.Errorf("%q.ReplaceAll(%q,%q) = %q; want %q",
				tc.pattern, tc.input, tc.replacement, actual, tc.output)
		}
	}
}

func TestReplaceAllLiteral(t *testing.T) {
	// Run ReplaceAll tests that do not have $ expansions.
	for _, tc := range replaceTests {
		if strings.Contains(tc.replacement, "$") {
			continue
		}
		re, err := Compile(tc.pattern)
		if err != nil {
			t.Errorf("Unexpected error compiling %q: %v", tc.pattern, err)
			continue
		}
		actual := re.ReplaceAllLiteralString(tc.input, tc.replacement)
		if actual != tc.output {
			t.Errorf("%q.ReplaceAllLiteralString(%q,%q) = %q; want %q",
				tc.pattern, tc.input, tc.replacement, actual, tc.output)
		}
		// now try bytes
		actual = string(re.ReplaceAllLiteral([]byte(tc.input), []byte(tc.replacement)))
		if actual != tc.output {
			t.Errorf("%q.ReplaceAllLiteral(%q,%q) = %q; want %q",
				tc.pattern, tc.input, tc.replacement, actual, tc.output)
		}
	}

	// Run literal-specific tests.
	for _, tc := range replaceLiteralTests {
		re, err := Compile(tc.pattern)
		if err != nil {
			t.Errorf("Unexpected error compiling %q: %v", tc.pattern, err)
			continue
		}
		actual := re.ReplaceAllLiteralString(tc.input, tc.replacement)
		if actual != tc.output {
			t.Errorf("%q.ReplaceAllLiteralString(%q,%q) = %q; want %q",
				tc.pattern, tc.input, tc.replacement, actual, tc.output)
		}
		// now try bytes
		actual = string(re.ReplaceAllLiteral([]byte(tc.input), []byte(tc.replacement)))
		if actual != tc.output {
			t.Errorf("%q.ReplaceAllLiteral(%q,%q) = %q; want %q",
				tc.pattern, tc.input, tc.replacement, actual, tc.output)
		}
	}
}

func TestReplaceAllFunc(t *testing.T) {
	for _, tc := range replaceFuncTests {
		re, err := Compile(tc.pattern)
		if err != nil {
			t.Errorf("Unexpected error compiling %q: %v", tc.pattern, err)
			continue
		}
		actual := re.ReplaceAllStringFunc(tc.input, tc.replacement)
		if actual != tc.output {
			t.Errorf("%q.ReplaceFunc(%q,%q) = %q; want %q",
				tc.pattern, tc.input, tc.replacement, actual, tc.output)
		}
		// now try bytes
		actual = string(re.ReplaceAllFunc([]byte(tc.input), func(s []byte) []byte { return []byte(tc.replacement(string(s))) }))
		if actual != tc.output {
			t.Errorf("%q.ReplaceFunc(%q,%q) = %q; want %q",
				tc.pattern, tc.input, tc.replacement, actual, tc.output)
		}
	}
}

type MetaTest struct {
	pattern, output, literal string
	isLiteral                bool
}

var metaTests = []MetaTest{
	{``, ``, ``, true},
	{`foo`, `foo`, `foo`, true},
	{`foo\.\$`, `foo\\\.\\\$`, `foo.$`, true}, // has meta but no operator
	{`foo.\$`, `foo\.\\\$`, `foo`, false},     // has escaped operators and real operators
	{`!@#$%^&*()_+-=[{]}\|,<.>/?~`, `!@#\$%\^&\*\(\)_\+-=\[\{\]\}\\\|,<\.>/\?~`, `!@#`, false},
}

func TestQuoteMeta(t *testing.T) {
	for _, tc := range metaTests {
		// Verify that QuoteMeta returns the expected string.
		quoted := QuoteMeta(tc.pattern)
		if quoted != tc.output {
			t.Errorf("QuoteMeta(`%s`) = `%s`; want `%s`",
				tc.pattern, quoted, tc.output)
			continue
		}

		// Verify that the quoted string is in fact treated as expected
		// by Compile -- i.e. that it matches the original, unquoted string.
		if tc.pattern != "" {
			re, err := Compile(quoted)
			if err != nil {
				t.Errorf("Unexpected error compiling QuoteMeta(`%s`): %v", tc.pattern, err)
				continue
			}
			src := "abc" + tc.pattern + "def"
			repl := "xyz"
			replaced := re.ReplaceAllString(src, repl)
			expected := "abcxyzdef"
			if replaced != expected {
				t.Errorf("QuoteMeta(`%s`).Replace(`%s`,`%s`) = `%s`; want `%s`",
					tc.pattern, src, repl, replaced, expected)
			}
		}
	}
}

func TestLiteralPrefix(t *testing.T) {
	for _, tc := range metaTests {
		// Literal method needs to scan the pattern.
		re := MustCompile(tc.pattern)
		str, complete := re.LiteralPrefix()
		if complete != tc.isLiteral {
			t.Errorf("LiteralPrefix(`%s`) = %t; want %t", tc.pattern, complete, tc.isLiteral)
		}
		if str != tc.literal {
			t.Errorf("LiteralPrefix(`%s`) = `%s`; want `%s`", tc.pattern, str, tc.literal)
		}
	}
}

type subexpCase struct {
	input string
	num   int
	names []string
}

var subexpCases = []subexpCase{
	{``, 0, nil},
	{`.*`, 0, nil},
	{`abba`, 0, nil},
	{`ab(b)a`, 1, []string{"", ""}},
	{`ab(.*)a`, 1, []string{"", ""}},
	{`(.*)ab(.*)a`, 2, []string{"", "", ""}},
	{`(.*)(ab)(.*)a`, 3, []string{"", "", "", ""}},
	{`(.*)((a)b)(.*)a`, 4, []string{"", "", "", "", ""}},
	{`(.*)(\(ab)(.*)a`, 3, []string{"", "", "", ""}},
	{`(.*)(\(a\)b)(.*)a`, 3, []string{"", "", "", ""}},
	{`(?P<foo>.*)(?P<bar>(a)b)(?P<foo>.*)a`, 4, []string{"", "foo", "bar", "", "foo"}},
}

func TestSubexp(t *testing.T) {
	for _, c := range subexpCases {
		re := MustCompile(c.input)
		n := re.NumSubexp()
		if n != c.num {
			t.Errorf("%q: NumSubexp = %d, want %d", c.input, n, c.num)
			continue
		}
		names := re.SubexpNames()
		if len(names) != 1+n {
			t.Errorf("%q: len(SubexpNames) = %d, want %d", c.input, len(names), n)
			continue
		}
		if c.names != nil {
			for i := 0; i < 1+n; i++ {
				if names[i] != c.names[i] {
					t.Errorf("%q: SubexpNames[%d] = %q, want %q", c.input, i, names[i], c.names[i])
				}
			}
		}
	}
}

var splitTests = []struct {
	s   string
	r   string
	n   int
	out []string
}{
	{"foo:and:bar", ":", -1, []string{"foo", "and", "bar"}},
	{"foo:and:bar", ":", 1, []string{"foo:and:bar"}},
	{"foo:and:bar", ":", 2, []string{"foo", "and:bar"}},
	{"foo:and:bar", "foo", -1, []string{"", ":and:bar"}},
	{"foo:and:bar", "bar", -1, []string{"foo:and:", ""}},
	{"foo:and:bar", "baz", -1, []string{"foo:and:bar"}},
	{"baabaab", "a", -1, []string{"b", "", "b", "", "b"}},
	{"baabaab", "a*", -1, []string{"b", "b", "b"}},
	{"baabaab", "ba*", -1, []string{"", "", "", ""}},
	{"foobar", "f*b*", -1, []string{"", "o", "o", "a", "r"}},
	{"foobar", "f+.*b+", -1, []string{"", "ar"}},
	{"foobooboar", "o{2}", -1, []string{"f", "b", "boar"}},
	{"a,b,c,d,e,f", ",", 3, []string{"a", "b", "c,d,e,f"}},
	{"a,b,c,d,e,f", ",", 0, nil},
	{",", ",", -1, []string{"", ""}},
	{",,,", ",", -1, []string{"", "", "", ""}},
	{"", ",", -1, []string{""}},
	{"", ".*", -1, []string{""}},
	{"", ".+", -1, []string{""}},
	{"", "", -1, []string{}},
	{"foobar", "", -1, []string{"f", "o", "o", "b", "a", "r"}},
	{"abaabaccadaaae", "a*", 5, []string{"", "b", "b", "c", "cadaaae"}},
	{":x:y:z:", ":", -1, []string{"", "x", "y", "z", ""}},
}

func TestSplit(t *testing.T) {
	for i, test := range splitTests {
		re, err := Compile(test.r)
		if err != nil {
			t.Errorf("#%d: %q: compile error: %s", i, test.r, err.Error())
			continue
		}

		split := re.Split(test.s, test.n)
		if !reflect.DeepEqual(split, test.out) {
			t.Errorf("#%d: %q: got %q; want %q", i, test.r, split, test.out)
		}

		if QuoteMeta(test.r) == test.r {
			strsplit := strings.SplitN(test.s, test.r, test.n)
			if !reflect.DeepEqual(split, strsplit) {
				t.Errorf("#%d: Split(%q, %q, %d): regexp vs strings mismatch\nregexp=%q\nstrings=%q", i, test.s, test.r, test.n, split, strsplit)
			}
		}
	}
}

func BenchmarkLiteral(b *testing.B) {
	x := strings.Repeat("x", 50) + "y"
	b.StopTimer()
	re := MustCompile("y")
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		if !re.MatchString(x) {
			b.Fatalf("no match!")
		}
	}
}

func BenchmarkNotLiteral(b *testing.B) {
	x := strings.Repeat("x", 50) + "y"
	b.StopTimer()
	re := MustCompile(".y")
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		if !re.MatchString(x) {
			b.Fatalf("no match!")
		}
	}
}

func BenchmarkMatchClass(b *testing.B) {
	b.StopTimer()
	x := strings.Repeat("xxxx", 20) + "w"
	re := MustCompile("[abcdw]")
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		if !re.MatchString(x) {
			b.Fatalf("no match!")
		}
	}
}

func BenchmarkMatchClass_InRange(b *testing.B) {
	b.StopTimer()
	// 'b' is between 'a' and 'c', so the charclass
	// range checking is no help here.
	x := strings.Repeat("bbbb", 20) + "c"
	re := MustCompile("[ac]")
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		if !re.MatchString(x) {
			b.Fatalf("no match!")
		}
	}
}

func BenchmarkReplaceAll(b *testing.B) {
	x := "abcdefghijklmnopqrstuvwxyz"
	b.StopTimer()
	re := MustCompile("[cjrw]")
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		re.ReplaceAllString(x, "")
	}
}

func BenchmarkAnchoredLiteralShortNonMatch(b *testing.B) {
	b.StopTimer()
	x := []byte("abcdefghijklmnopqrstuvwxyz")
	re := MustCompile("^zbc(d|e)")
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		re.Match(x)
	}
}

func BenchmarkAnchoredLiteralLongNonMatch(b *testing.B) {
	b.StopTimer()
	x := []byte("abcdefghijklmnopqrstuvwxyz")
	for i := 0; i < 15; i++ {
		x = append(x, x...)
	}
	re := MustCompile("^zbc(d|e)")
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		re.Match(x)
	}
}

func BenchmarkAnchoredShortMatch(b *testing.B) {
	b.StopTimer()
	x := []byte("abcdefghijklmnopqrstuvwxyz")
	re := MustCompile("^.bc(d|e)")
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		re.Match(x)
	}
}

func BenchmarkAnchoredLongMatch(b *testing.B) {
	b.StopTimer()
	x := []byte("abcdefghijklmnopqrstuvwxyz")
	for i := 0; i < 15; i++ {
		x = append(x, x...)
	}
	re := MustCompile("^.bc(d|e)")
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		re.Match(x)
	}
}
