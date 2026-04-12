// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package objabi

import (
	"slices"
	"testing"
)

func TestParseArgs(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name  string
		input string
		want  []string
	}{
		// GCC-compatibility test cases from test-expandargv.c
		// Source code: https://github.com/gcc-mirror/gcc/blob/releases/gcc-15.2.0/libiberty/testsuite/test-expandargv.c#L72
		{`crlf`, "a\r\nb", []string{"a", "b"}},                                       // test 0
		{"newline", "a\nb", []string{"a", "b"}},                                      // test 1
		{"null byte in arg", "a\x00b", []string{"a\x00b"}},                           // test 2: GCC parser gives ["a"]
		{"null byte only", "\x00", []string{"\x00"}},                                 // test 3: GCC parser gives []
		{"leading newline", "\na\nb", []string{"a", "b"}},                            // test 4
		{"empty quotes", "a\n''\nb", []string{"a", "", "b"}},                         // test 5
		{"quoted newlines", "a\n'a\n\nb'\nb", []string{"a", "a\n\nb", "b"}},          // test 6
		{"single quote no escapes", "'a\\$VAR' '\\\"'", []string{"a\\$VAR", "\\\""}}, // test 7
		{"line continuation", "\"ab\\\ncd\" ef\\\ngh", []string{"abcd", "efgh"}},     // test 8
		// test 8.1 (additional verification for Windows line separators)
		{"line continuation crlf", "\"ab\\\r\ncd\" ef\\\r\ngh", []string{"abcd", "efgh"}},
		{"double quote escapes", "\"\\$VAR\" \"\\`\" \"\\\"\" \"\\\\\" \"\\n\" \"\\t\"",
			[]string{"$VAR", "`", `"`, `\`, `\n`, `\t`}}, // test 9
		{"whitespace only", "\t \n \t ", nil}, // test 10
		{"single space", " ", nil},            // test 11
		{"multiple spaces", "   ", nil},       // test 12

		// Additional edge cases for peace of mind
		{"basic split", "a b c", []string{"a", "b", "c"}},
		{"tabs", "a\tb\tc", []string{"a", "b", "c"}},
		{"mixed quotes", `a 'b c' "d e"`, []string{"a", "b c", "d e"}},
		{"adjacent quotes", `'a'"b"`, []string{"ab"}}, // no whitespace - no split
		{"empty input", "", nil},
		{"empty single quotes", "''", []string{""}},
		{"empty double quotes", `""`, []string{""}},
		{"nested quotes in single", `'"hello"'`, []string{`"hello"`}},
		{"nested quotes in double", `"'hello'"`, []string{"'hello'"}},
		// GCC-specific (differs from LLVM): backslash outside quotes escapes the next character
		{"backslash escape outside quotes", `\abc`, []string{"abc"}},
		{"trailing backslash", `abc\`, []string{"abc"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ParseArgs([]byte(tt.input))
			if !slices.Equal(got, tt.want) {
				t.Errorf("parseArgs(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}
