// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"exec"
	"go/ast"
	"go/parser"
	"go/printer"
	"io/ioutil"
	"os"
	"testing"
)

type testCase struct {
	Name string
	Fn   func(*ast.File) bool
	In   string
	Out  string
}

var testCases []testCase

func addTestCases(t []testCase) {
	testCases = append(testCases, t...)
}

func parseFixPrint(t *testing.T, fn func(*ast.File) bool, desc, in string) (out string, fixed, ok bool) {
	file, err := parser.ParseFile(fset, desc, in, parserMode)
	if err != nil {
		t.Errorf("%s: parsing: %v", desc, err)
		return
	}

	var buf bytes.Buffer
	buf.Reset()
	_, err = (&printer.Config{printerMode, tabWidth}).Fprint(&buf, fset, file)
	if err != nil {
		t.Errorf("%s: printing: %v", desc, err)
		return
	}
	if s := buf.String(); in != s {
		t.Errorf("%s: not gofmt-formatted.\n--- %s\n%s\n--- %s | gofmt\n%s",
			desc, desc, in, desc, s)
		tdiff(t, in, s)
		return
	}

	if fn == nil {
		for _, fix := range fixes {
			if fix.f(file) {
				fixed = true
			}
		}
	} else {
		fixed = fn(file)
	}

	buf.Reset()
	_, err = (&printer.Config{printerMode, tabWidth}).Fprint(&buf, fset, file)
	if err != nil {
		t.Errorf("%s: printing: %v", desc, err)
		return
	}

	return buf.String(), fixed, true
}

func TestRewrite(t *testing.T) {
	for _, tt := range testCases {
		// Apply fix: should get tt.Out.
		out, fixed, ok := parseFixPrint(t, tt.Fn, tt.Name, tt.In)
		if !ok {
			continue
		}

		if out != tt.Out {
			t.Errorf("%s: incorrect output.\n--- have\n%s\n--- want\n%s", tt.Name, out, tt.Out)
			tdiff(t, out, tt.Out)
			continue
		}

		if changed := out != tt.In; changed != fixed {
			t.Errorf("%s: changed=%v != fixed=%v", tt.Name, changed, fixed)
			continue
		}

		// Should not change if run again.
		out2, fixed2, ok := parseFixPrint(t, tt.Fn, tt.Name+" output", out)
		if !ok {
			continue
		}

		if fixed2 {
			t.Errorf("%s: applied fixes during second round", tt.Name)
			continue
		}

		if out2 != out {
			t.Errorf("%s: changed output after second round of fixes.\n--- output after first round\n%s\n--- output after second round\n%s",
				tt.Name, out, out2)
			tdiff(t, out, out2)
		}
	}
}

func tdiff(t *testing.T, a, b string) {
	f1, err := ioutil.TempFile("", "gofix")
	if err != nil {
		t.Error(err)
		return
	}
	defer os.Remove(f1.Name())
	defer f1.Close()

	f2, err := ioutil.TempFile("", "gofix")
	if err != nil {
		t.Error(err)
		return
	}
	defer os.Remove(f2.Name())
	defer f2.Close()

	f1.Write([]byte(a))
	f2.Write([]byte(b))

	diffcmd, err := exec.LookPath("diff")
	if err != nil {
		t.Error(err)
		return
	}

	c, err := exec.Run(diffcmd, []string{"diff", f1.Name(), f2.Name()}, nil, "",
		exec.DevNull, exec.Pipe, exec.MergeWithStdout)
	if err != nil {
		t.Error(err)
		return
	}
	defer c.Close()

	data, err := ioutil.ReadAll(c.Stdout)
	if err != nil {
		t.Error(err)
		return
	}

	t.Error(string(data))
}
