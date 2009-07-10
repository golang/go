// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package parser

import (
	"go/ast";
	"go/parser";
	"os";
	"testing";
)


var illegalInputs = []interface{} {
	nil,
	3.14,
	[]byte(nil),
	"foo!",
}


func TestParseIllegalInputs(t *testing.T) {
	for _, src := range illegalInputs {
		prog, err := Parse(src, 0);
		if err == nil {
			t.Errorf("Parse(%v) should have failed", src);
		}
	}
}


var validPrograms = []interface{} {
	`package main`,
	`package main import "fmt" func main() { fmt.Println("Hello, World!") }`,
}


func TestParseValidPrograms(t *testing.T) {
	for _, src := range validPrograms {
		prog, err := Parse(src, 0);
		if err != nil {
			t.Errorf("Parse(%q) failed: %v", src, err);
		}
	}
}


var validFiles = []string {
	"parser.go",
	"parser_test.go",
}


func TestParse3(t *testing.T) {
	for _, filename := range validFiles {
		src, err := os.Open(filename, os.O_RDONLY, 0);
		defer src.Close();
		if err != nil {
			t.Fatal(err);
		}

		prog, err := Parse(src, 0);
		if err != nil {
			t.Errorf("Parse(%s): %v", filename, err);
		}
	}
}
