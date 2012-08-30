// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test string literal syntax.

package main

import "os"

var ecode int

func assert(a, b, c string) {
	if a != b {
		ecode = 1
		print("FAIL: ", c, ": ", a, "!=", b, "\n")
		var max int = len(a)
		if len(b) > max {
			max = len(b)
		}
		for i := 0; i < max; i++ {
			ac := 0
			bc := 0
			if i < len(a) {
				ac = int(a[i])
			}
			if i < len(b) {
				bc = int(b[i])
			}
			if ac != bc {
				print("\ta[", i, "] = ", ac, "; b[", i, "] =", bc, "\n")
			}
		}
	}
}

const (
	gx1    = "aä本☺"
	gx2    = "aä\xFF\xFF本☺"
	gx2fix = "aä\uFFFD\uFFFD本☺"
)

var (
	gr1 = []rune(gx1)
	gr2 = []rune(gx2)
	gb1 = []byte(gx1)
	gb2 = []byte(gx2)
)

func main() {
	ecode = 0
	s :=
		"" +
			" " +
			"'`" +
			"a" +
			"ä" +
			"本" +
			"\a\b\f\n\r\t\v\\\"" +
			"\000\123\x00\xca\xFE\u0123\ubabe\U0000babe" +

			`` +
			` ` +
			`'"` +
			`a` +
			`ä` +
			`本` +
			`\a\b\f\n\r\t\v\\\'` +
			`\000\123\x00\xca\xFE\u0123\ubabe\U0000babe` +
			`\x\u\U\`

	assert("", ``, "empty")
	assert(" ", " ", "blank")
	assert("\x61", "a", "lowercase a")
	assert("\x61", `a`, "lowercase a (backquote)")
	assert("\u00e4", "ä", "a umlaut")
	assert("\u00e4", `ä`, "a umlaut (backquote)")
	assert("\u672c", "本", "nihon")
	assert("\u672c", `本`, "nihon (backquote)")
	assert("\x07\x08\x0c\x0a\x0d\x09\x0b\x5c\x22",
		"\a\b\f\n\r\t\v\\\"",
		"backslashes")
	assert("\\a\\b\\f\\n\\r\\t\\v\\\\\\\"",
		`\a\b\f\n\r\t\v\\\"`,
		"backslashes (backquote)")
	assert("\x00\x53\000\xca\376S몾몾",
		"\000\123\x00\312\xFE\u0053\ubabe\U0000babe",
		"backslashes 2")
	assert("\\000\\123\\x00\\312\\xFE\\u0123\\ubabe\\U0000babe",
		`\000\123\x00\312\xFE\u0123\ubabe\U0000babe`,
		"backslashes 2 (backquote)")
	assert("\\x\\u\\U\\", `\x\u\U\`, "backslash 3 (backquote)")

	// test large and surrogate-half runes. perhaps not the most logical place for these tests.
	var r int32
	r = 0x10ffff // largest rune value
	s = string(r)
	assert(s, "\xf4\x8f\xbf\xbf", "largest rune")
	r = 0x10ffff + 1
	s = string(r)
	assert(s, "\xef\xbf\xbd", "too-large rune")
	r = 0xD800
	s = string(r)
	assert(s, "\xef\xbf\xbd", "surrogate rune min")
	r = 0xDFFF
	s = string(r)
	assert(s, "\xef\xbf\xbd", "surrogate rune max")
	r = -1
	s = string(r)
	assert(s, "\xef\xbf\xbd", "negative rune")
	
	// the large rune tests again, this time using constants instead of a variable.
	// these conversions will be done at compile time.
	s = string(0x10ffff) // largest rune value
	assert(s, "\xf4\x8f\xbf\xbf", "largest rune constant")
	s = string(0x10ffff + 1)
	assert(s, "\xef\xbf\xbd", "too-large rune constant")
	s = string(0xD800)
	assert(s, "\xef\xbf\xbd", "surrogate rune min constant")
	s = string(0xDFFF)
	assert(s, "\xef\xbf\xbd", "surrogate rune max constant")
	s = string(-1)
	assert(s, "\xef\xbf\xbd", "negative rune")

	assert(string(gr1), gx1, "global ->[]rune")
	assert(string(gr2), gx2fix, "global invalid ->[]rune")
	assert(string(gb1), gx1, "->[]byte")
	assert(string(gb2), gx2, "global invalid ->[]byte")

	var (
		r1 = []rune(gx1)
		r2 = []rune(gx2)
		b1 = []byte(gx1)
		b2 = []byte(gx2)
	)
	assert(string(r1), gx1, "->[]rune")
	assert(string(r2), gx2fix, "invalid ->[]rune")
	assert(string(b1), gx1, "->[]byte")
	assert(string(b2), gx2, "invalid ->[]byte")

	os.Exit(ecode)
}
