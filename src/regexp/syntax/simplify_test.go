// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import "testing"

var simplifyTests = []struct {
	Regexp string
	Simple string
}{
	// Already-simple constructs
	{`a`, `a`},
	{`ab`, `ab`},
	{`a|b`, `[a-b]`},
	{`ab|cd`, `ab|cd`},
	{`(ab)*`, `(ab)*`},
	{`(ab)+`, `(ab)+`},
	{`(ab)?`, `(ab)?`},
	{`.`, `(?s:.)`},
	{`^`, `(?m:^)`},
	{`$`, `(?m:$)`},
	{`[ac]`, `[ac]`},
	{`[^ac]`, `[^ac]`},

	// Posix character classes
	{`[[:alnum:]]`, `[0-9A-Za-z]`},
	{`[[:alpha:]]`, `[A-Za-z]`},
	{`[[:blank:]]`, `[\t ]`},
	{`[[:cntrl:]]`, `[\x00-\x1f\x7f]`},
	{`[[:digit:]]`, `[0-9]`},
	{`[[:graph:]]`, `[!-~]`},
	{`[[:lower:]]`, `[a-z]`},
	{`[[:print:]]`, `[ -~]`},
	{`[[:punct:]]`, "[!-/:-@\\[-`\\{-~]"},
	{`[[:space:]]`, `[\t-\r ]`},
	{`[[:upper:]]`, `[A-Z]`},
	{`[[:xdigit:]]`, `[0-9A-Fa-f]`},

	// Perl character classes
	{`\d`, `[0-9]`},
	{`\s`, `[\t-\n\f-\r ]`},
	{`\w`, `[0-9A-Z_a-z]`},
	{`\D`, `[^0-9]`},
	{`\S`, `[^\t-\n\f-\r ]`},
	{`\W`, `[^0-9A-Z_a-z]`},
	{`[\d]`, `[0-9]`},
	{`[\s]`, `[\t-\n\f-\r ]`},
	{`[\w]`, `[0-9A-Z_a-z]`},
	{`[\D]`, `[^0-9]`},
	{`[\S]`, `[^\t-\n\f-\r ]`},
	{`[\W]`, `[^0-9A-Z_a-z]`},

	// Posix repetitions
	{`a{1}`, `a`},
	{`a{2}`, `aa`},
	{`a{5}`, `aaaaa`},
	{`a{0,1}`, `a?`},
	// The next three are illegible because Simplify inserts (?:)
	// parens instead of () parens to avoid creating extra
	// captured subexpressions.  The comments show a version with fewer parens.
	{`(a){0,2}`, `(?:(a)(a)?)?`},                       //       (aa?)?
	{`(a){0,4}`, `(?:(a)(?:(a)(?:(a)(a)?)?)?)?`},       //   (a(a(aa?)?)?)?
	{`(a){2,6}`, `(a)(a)(?:(a)(?:(a)(?:(a)(a)?)?)?)?`}, // aa(a(a(aa?)?)?)?
	{`a{0,2}`, `(?:aa?)?`},                             //       (aa?)?
	{`a{0,4}`, `(?:a(?:a(?:aa?)?)?)?`},                 //   (a(a(aa?)?)?)?
	{`a{2,6}`, `aa(?:a(?:a(?:aa?)?)?)?`},               // aa(a(a(aa?)?)?)?
	{`a{0,}`, `a*`},
	{`a{1,}`, `a+`},
	{`a{2,}`, `aa+`},
	{`a{5,}`, `aaaaa+`},

	// Test that operators simplify their arguments.
	{`(?:a{1,}){1,}`, `a+`},
	{`(a{1,}b{1,})`, `(a+b+)`},
	{`a{1,}|b{1,}`, `a+|b+`},
	{`(?:a{1,})*`, `(?:a+)*`},
	{`(?:a{1,})+`, `a+`},
	{`(?:a{1,})?`, `(?:a+)?`},
	{``, `(?:)`},
	{`a{0}`, `(?:)`},

	// Character class simplification
	{`[ab]`, `[a-b]`},
	{`[a-za-za-z]`, `[a-z]`},
	{`[A-Za-zA-Za-z]`, `[A-Za-z]`},
	{`[ABCDEFGH]`, `[A-H]`},
	{`[AB-CD-EF-GH]`, `[A-H]`},
	{`[W-ZP-XE-R]`, `[E-Z]`},
	{`[a-ee-gg-m]`, `[a-m]`},
	{`[a-ea-ha-m]`, `[a-m]`},
	{`[a-ma-ha-e]`, `[a-m]`},
	{`[a-zA-Z0-9 -~]`, `[ -~]`},

	// Empty character classes
	{`[^[:cntrl:][:^cntrl:]]`, `[^\x00-\x{10FFFF}]`},

	// Full character classes
	{`[[:cntrl:][:^cntrl:]]`, `(?s:.)`},

	// Unicode case folding.
	{`(?i)A`, `(?i:A)`},
	{`(?i)a`, `(?i:A)`},
	{`(?i)[A]`, `(?i:A)`},
	{`(?i)[a]`, `(?i:A)`},
	{`(?i)K`, `(?i:K)`},
	{`(?i)k`, `(?i:K)`},
	{`(?i)\x{212a}`, "(?i:K)"},
	{`(?i)[K]`, "[Kk\u212A]"},
	{`(?i)[k]`, "[Kk\u212A]"},
	{`(?i)[\x{212a}]`, "[Kk\u212A]"},
	{`(?i)[a-z]`, "[A-Za-z\u017F\u212A]"},
	{`(?i)[\x00-\x{FFFD}]`, "[\\x00-\uFFFD]"},
	{`(?i)[\x00-\x{10FFFF}]`, `(?s:.)`},

	// Empty string as a regular expression.
	// The empty string must be preserved inside parens in order
	// to make submatches work right, so these tests are less
	// interesting than they might otherwise be.  String inserts
	// explicit (?:) in place of non-parenthesized empty strings,
	// to make them easier to spot for other parsers.
	{`(a|b|)`, `([a-b]|(?:))`},
	{`(|)`, `()`},
	{`a()`, `a()`},
	{`(()|())`, `(()|())`},
	{`(a|)`, `(a|(?:))`},
	{`ab()cd()`, `ab()cd()`},
	{`()`, `()`},
	{`()*`, `()*`},
	{`()+`, `()+`},
	{`()?`, `()?`},
	{`(){0}`, `(?:)`},
	{`(){1}`, `()`},
	{`(){1,}`, `()+`},
	{`(){0,2}`, `(?:()()?)?`},
}

func TestSimplify(t *testing.T) {
	for _, tt := range simplifyTests {
		re, err := Parse(tt.Regexp, MatchNL|Perl&^OneLine)
		if err != nil {
			t.Errorf("Parse(%#q) = error %v", tt.Regexp, err)
			continue
		}
		s := re.Simplify().String()
		if s != tt.Simple {
			t.Errorf("Simplify(%#q) = %#q, want %#q", tt.Regexp, s, tt.Simple)
		}
	}
}
