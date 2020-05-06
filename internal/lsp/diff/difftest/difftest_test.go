// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package difftest supplies a set of tests that will operate on any
// implementation of a diff algorithm as exposed by
// "golang.org/x/tools/internal/lsp/diff"
package difftest_test

import (
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"strings"
	"testing"

	"golang.org/x/tools/internal/lsp/diff/difftest"
	"golang.org/x/tools/internal/testenv"
)

func TestVerifyUnified(t *testing.T) {
	testenv.NeedsTool(t, "diff")
	for _, test := range difftest.TestCases {
		t.Run(test.Name, func(t *testing.T) {
			t.Helper()
			if test.NoDiff {
				t.Skip("diff tool produces expected different results")
			}
			diff, err := getDiffOutput(test.In, test.Out)
			if err != nil {
				t.Fatal(err)
			}
			if len(diff) > 0 {
				diff = difftest.UnifiedPrefix + diff
			}
			if diff != test.Unified {
				t.Errorf("unified:\n%q\ndiff -u:\n%q", test.Unified, diff)
			}
		})
	}
}

func getDiffOutput(a, b string) (string, error) {
	fileA, err := ioutil.TempFile("", "myers.in")
	if err != nil {
		return "", err
	}
	defer os.Remove(fileA.Name())
	if _, err := fileA.Write([]byte(a)); err != nil {
		return "", err
	}
	if err := fileA.Close(); err != nil {
		return "", err
	}
	fileB, err := ioutil.TempFile("", "myers.in")
	if err != nil {
		return "", err
	}
	defer os.Remove(fileB.Name())
	if _, err := fileB.Write([]byte(b)); err != nil {
		return "", err
	}
	if err := fileB.Close(); err != nil {
		return "", err
	}
	cmd := exec.Command("diff", "-u", fileA.Name(), fileB.Name())
	out, err := cmd.CombinedOutput()
	if err != nil {
		if _, ok := err.(*exec.ExitError); !ok {
			return "", fmt.Errorf("failed to run diff -u %v %v: %v\n%v", fileA.Name(), fileB.Name(), err, string(out))
		}
	}
	diff := string(out)
	if len(diff) <= 0 {
		return diff, nil
	}
	bits := strings.SplitN(diff, "\n", 3)
	if len(bits) != 3 {
		return "", fmt.Errorf("diff output did not have file prefix:\n%s", diff)
	}
	return bits[2], nil
}
