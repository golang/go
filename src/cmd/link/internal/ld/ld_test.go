// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"internal/testenv"
	"io/ioutil"
	"os"
	"os/exec"
	"strings"
	"testing"
)

func TestUndefinedRelocErrors(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	dir, err := ioutil.TempDir("", "go-build")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

	out, err := exec.Command(testenv.GoToolPath(t), "build", "./testdata/issue10978").CombinedOutput()
	if err == nil {
		t.Fatal("expected build to fail")
	}

	wantErrors := map[string]int{
		// Main function has dedicated error message.
		"function main is undeclared in the main package": 1,

		// Single error reporting per each symbol.
		// This way, duplicated messages are not reported for
		// multiple relocations with a same name.
		"main.defined1: relocation target main.undefined not defined": 1,
		"main.defined2: relocation target main.undefined not defined": 1,
	}
	unexpectedErrors := map[string]int{}

	for _, l := range strings.Split(string(out), "\n") {
		if strings.HasPrefix(l, "#") || l == "" {
			continue
		}
		matched := ""
		for want := range wantErrors {
			if strings.Contains(l, want) {
				matched = want
				break
			}
		}
		if matched != "" {
			wantErrors[matched]--
		} else {
			unexpectedErrors[l]++
		}
	}

	for want, n := range wantErrors {
		switch {
		case n > 0:
			t.Errorf("unmatched error: %s (x%d)", want, n)
		case n < 0:
			t.Errorf("extra errors: %s (x%d)", want, -n)
		}
	}
	for unexpected, n := range unexpectedErrors {
		t.Errorf("unexpected error: %s (x%d)", unexpected, n)
	}
}
