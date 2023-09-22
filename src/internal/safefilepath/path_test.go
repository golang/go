// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package safefilepath_test

import (
	"internal/safefilepath"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

type PathTest struct {
	path, result string
}

const invalid = ""

var fspathtests = []PathTest{
	{".", "."},
	{"/a/b/c", "/a/b/c"},
	{"a\x00b", invalid},
}

var winreservedpathtests = []PathTest{
	{`a\b`, `a\b`},
	{`a:b`, `a:b`},
	{`a/b:c`, `a/b:c`},
	{`NUL`, `NUL`},
	{`./com1`, `./com1`},
	{`a/nul/b`, `a/nul/b`},
}

// Whether a reserved name with an extension is reserved or not varies by
// Windows version.
var winreservedextpathtests = []PathTest{
	{"nul.txt", "nul.txt"},
	{"a/nul.txt/b", "a/nul.txt/b"},
}

var plan9reservedpathtests = []PathTest{
	{`#c`, `#c`},
}

func TestFromFS(t *testing.T) {
	switch runtime.GOOS {
	case "windows":
		if canWriteFile(t, "NUL") {
			t.Errorf("can unexpectedly write a file named NUL on Windows")
		}
		if canWriteFile(t, "nul.txt") {
			fspathtests = append(fspathtests, winreservedextpathtests...)
		} else {
			winreservedpathtests = append(winreservedpathtests, winreservedextpathtests...)
		}
		for i := range winreservedpathtests {
			winreservedpathtests[i].result = invalid
		}
		for i := range fspathtests {
			fspathtests[i].result = filepath.FromSlash(fspathtests[i].result)
		}
	case "plan9":
		for i := range plan9reservedpathtests {
			plan9reservedpathtests[i].result = invalid
		}
	}
	tests := fspathtests
	tests = append(tests, winreservedpathtests...)
	tests = append(tests, plan9reservedpathtests...)
	for _, test := range tests {
		got, err := safefilepath.FromFS(test.path)
		if (got == "") != (err != nil) {
			t.Errorf(`FromFS(%q) = %q, %v; want "" only if err != nil`, test.path, got, err)
		}
		if got != test.result {
			t.Errorf("FromFS(%q) = %q, %v; want %q", test.path, got, err, test.result)
		}
	}
}

func canWriteFile(t *testing.T, name string) bool {
	path := filepath.Join(t.TempDir(), name)
	os.WriteFile(path, []byte("ok"), 0666)
	b, _ := os.ReadFile(path)
	return string(b) == "ok"
}
