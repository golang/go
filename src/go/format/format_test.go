// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package format

import (
	"bytes"
	"go/parser"
	"go/token"
	"io/ioutil"
	"strings"
	"testing"
)

const testfile = "format_test.go"

func diff(t *testing.T, dst, src []byte) {
	line := 1
	offs := 0 // line offset
	for i := 0; i < len(dst) && i < len(src); i++ {
		d := dst[i]
		s := src[i]
		if d != s {
			t.Errorf("dst:%d: %s\n", line, dst[offs:i+1])
			t.Errorf("src:%d: %s\n", line, src[offs:i+1])
			return
		}
		if s == '\n' {
			line++
			offs = i + 1
		}
	}
	if len(dst) != len(src) {
		t.Errorf("len(dst) = %d, len(src) = %d\nsrc = %q", len(dst), len(src), src)
	}
}

func TestNode(t *testing.T) {
	src, err := ioutil.ReadFile(testfile)
	if err != nil {
		t.Fatal(err)
	}

	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, testfile, src, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer

	if err = Node(&buf, fset, file); err != nil {
		t.Fatal("Node failed:", err)
	}

	diff(t, buf.Bytes(), src)
}

func TestSource(t *testing.T) {
	src, err := ioutil.ReadFile(testfile)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Source(src)
	if err != nil {
		t.Fatal("Source failed:", err)
	}

	diff(t, res, src)
}

// Test cases that are expected to fail are marked by the prefix "ERROR".
var tests = []string{
	// declaration lists
	`import "go/format"`,
	"var x int",
	"var x int\n\ntype T struct{}",

	// statement lists
	"x := 0",
	"f(a, b, c)\nvar x int = f(1, 2, 3)",

	// indentation, leading and trailing space
	"\tx := 0\n\tgo f()",
	"\tx := 0\n\tgo f()\n\n\n",
	"\n\t\t\n\n\tx := 0\n\tgo f()\n\n\n",
	"\n\t\t\n\n\t\t\tx := 0\n\t\t\tgo f()\n\n\n",
	"\n\t\t\n\n\t\t\tx := 0\n\t\t\tconst s = `\nfoo\n`\n\n\n", // no indentation inside raw strings

	// erroneous programs
	"ERROR1 + 2 +",
	"ERRORx :=  0",
}

func String(s string) (string, error) {
	res, err := Source([]byte(s))
	if err != nil {
		return "", err
	}
	return string(res), nil
}

func TestPartial(t *testing.T) {
	for _, src := range tests {
		if strings.HasPrefix(src, "ERROR") {
			// test expected to fail
			src = src[5:] // remove ERROR prefix
			res, err := String(src)
			if err == nil && res == src {
				t.Errorf("formatting succeeded but was expected to fail:\n%q", src)
			}
		} else {
			// test expected to succeed
			res, err := String(src)
			if err != nil {
				t.Errorf("formatting failed (%s):\n%q", err, src)
			} else if res != src {
				t.Errorf("formatting incorrect:\nsource: %q\nresult: %q", src, res)
			}
		}
	}
}
