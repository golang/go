// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main_test

import (
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"
)

var godocTests = []struct {
	args      []string
	matches   []string // regular expressions
	dontmatch []string // regular expressions
}{
	{
		args: []string{"fmt"},
		matches: []string{
			`import "fmt"`,
			`Package fmt implements formatted I/O`,
		},
	},
	{
		args: []string{"io", "WriteString"},
		matches: []string{
			`func WriteString\(`,
			`WriteString writes the contents of the string s to w`,
		},
	},
	{
		args: []string{"nonexistingpkg"},
		matches: []string{
			`no such file or directory|does not exist`,
		},
	},
	{
		args: []string{"fmt", "NonexistentSymbol"},
		matches: []string{
			`No match found\.`,
		},
	},
	{
		args: []string{"-src", "syscall", "Open"},
		matches: []string{
			`func Open\(`,
		},
		dontmatch: []string{
			`No match found\.`,
		},
	},
}

// Basic regression test for godoc command-line tool.
func TestGodoc(t *testing.T) {
	tmp, err := ioutil.TempDir("", "godoc-regtest-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmp)

	bin := filepath.Join(tmp, "godoc")
	if runtime.GOOS == "windows" {
		bin += ".exe"
	}
	cmd := exec.Command("go", "build", "-o", bin)
	if err := cmd.Run(); err != nil {
		t.Fatalf("Building godoc: %v", err)
	}

	for _, test := range godocTests {
		cmd := exec.Command(bin, test.args...)
		cmd.Args[0] = "godoc"
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Errorf("Running with args %#v: %v", test.args, err)
			continue
		}
		for _, pat := range test.matches {
			re := regexp.MustCompile(pat)
			if !re.Match(out) {
				t.Errorf("godoc %v =\n%s\nwanted /%v/", strings.Join(test.args, " "), out, pat)
			}
		}
		for _, pat := range test.dontmatch {
			re := regexp.MustCompile(pat)
			if re.Match(out) {
				t.Errorf("godoc %v =\n%s\ndid not want /%v/", strings.Join(test.args, " "), out, pat)
			}
		}
	}
}
