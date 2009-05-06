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


func TestParse0(t *testing.T) {
	// test nil []bytes source
	var src []byte;
	prog, ok := Parse(src, nil, 0);
	if ok {
		t.Errorf("parse should have failed");
	}
}


func TestParse1(t *testing.T) {
	// test string source
	src := `package main import "fmt" func main() { fmt.Println("Hello, World!") }`;
	prog, ok := Parse(src, nil, 0);
	if !ok {
		t.Errorf("parse failed");
	}
}

func TestParse2(t *testing.T) {
	// test io.Read source
	filename := "parser_test.go";
	src, err := os.Open(filename, os.O_RDONLY, 0);
	defer src.Close();
	if err != nil {
		t.Errorf("cannot open %s (%s)\n", filename, err.String());
	}

	prog, ok := Parse(src, nil, 0);
	if !ok {
		t.Errorf("parse failed");
	}
}
