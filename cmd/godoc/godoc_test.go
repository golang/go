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
	"testing"
)

var godocTests = []struct {
	args    []string
	matches []string // regular expressions
}{
	{
		[]string{"fmt"},
		[]string{
			`import "fmt"`,
			`Package fmt implements formatted I/O`,
		},
	},
	{
		[]string{"io", "WriteString"},
		[]string{
			`import "io"`,
			`func WriteString\(`,
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
		logged := false
		for _, pat := range test.matches {
			re := regexp.MustCompile(pat)
			if !re.Match(out) {
				if !logged {
					t.Logf("Output of running with args %#v:\n%s", test.args, out)
					logged = true
				}
				t.Errorf("Did not match /%v/", pat)
			}
		}
	}
}
