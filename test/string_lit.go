// $G $F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "os"

var ecode int;

func assert(a, b, c string) {
	if a != b {
		ecode = 1;
		print("FAIL: ", c, ": ", a, "!=", b, "\n");
		var max int = len(a);
		if len(b) > max {
			max = len(b);
		}
		for i := 0; i < max; i++ {
			ac := 0;
			bc := 0;
			if i < len(a) {
				ac = int(a[i]);
			}
			if i < len(b) {
				bc = int(b[i]);
			}
			if ac != bc {
				print("\ta[", i, "] = ", ac, "; b[", i, "] =", bc, "\n");
			}
		}
	}
}

func main() {
	ecode = 0;
	s :=
		""
		" "
		"'`"
		"a"
		"ä"
		"本"
		"\a\b\f\n\r\t\v\\\""
		"\000\123\x00\xca\xFE\u0123\ubabe\U0000babe"

		``
		` `
		`'"`
		`a`
		`ä`
		`本`
		`\a\b\f\n\r\t\v\\\'`
		`\000\123\x00\xca\xFE\u0123\ubabe\U0000babe`
		`\x\u\U\`
	;
	assert("", ``, "empty");
	assert(" ", " ", "blank");
	assert("\x61", "a", "lowercase a");
	assert("\x61", `a`, "lowercase a (backquote)");
	assert("\u00e4", "ä", "a umlaut");
	assert("\u00e4", `ä`, "a umlaut (backquote)");
	assert("\u672c", "本", "nihon");
	assert("\u672c", `本`, "nihon (backquote)");
	assert("\x07\x08\x0c\x0a\x0d\x09\x0b\x5c\x22",
	       "\a\b\f\n\r\t\v\\\"",
	       "backslashes");
	assert("\\a\\b\\f\\n\\r\\t\\v\\\\\\\"",
		`\a\b\f\n\r\t\v\\\"`,
	       "backslashes (backquote)");
	assert("\x00\x53\000\xca\376S몾몾",
		"\000\123\x00\312\xFE\u0053\ubabe\U0000babe",
		"backslashes 2");
	assert("\\000\\123\\x00\\312\\xFE\\u0123\\ubabe\\U0000babe",
		`\000\123\x00\312\xFE\u0123\ubabe\U0000babe`,
		"backslashes 2 (backquote)");
	assert("\\x\\u\\U\\", `\x\u\U\`, "backslash 3 (backquote)");

	// test large runes. perhaps not the most logical place for this test.
	var r int32;
	r = 0x10ffff;	// largest rune value
	s = string(r);
	assert(s, "\xf4\x8f\xbf\xbf", "largest rune");
	r = 0x10ffff + 1;
	s = string(r);
	assert(s, "\xef\xbf\xbd", "too-large rune");
	os.Exit(ecode);
}
