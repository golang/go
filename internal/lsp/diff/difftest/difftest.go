// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package difftest supplies a set of tests that will operate on any
// implementation of a diff algorithm as exposed by
// "golang.org/x/tools/internal/lsp/diff"
package difftest

import (
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"strings"
	"testing"

	"golang.org/x/tools/internal/lsp/diff"
	"golang.org/x/tools/internal/span"
)

const (
	fileA         = "from"
	fileB         = "to"
	unifiedPrefix = "--- " + fileA + "\n+++ " + fileB + "\n"
)

var verifyDiff = flag.Bool("verify-diff", false, "Check that the unified diff output matches `diff -u`")

func DiffTest(t *testing.T, compute diff.ComputeEdits) {
	t.Helper()
	for _, test := range []struct {
		name, in, out, unified string
		nodiff                 bool
	}{{
		name: "empty",
		in:   "",
		out:  "",
	}, {
		name: "no_diff",
		in:   "gargantuan\n",
		out:  "gargantuan\n",
	}, {
		name: "replace_all",
		in:   "gord\n",
		out:  "gourd\n",
		unified: unifiedPrefix + `
@@ -1 +1 @@
-gord
+gourd
`[1:],
	}, {
		name: "insert_rune",
		in:   "gord\n",
		out:  "gourd\n",
		unified: unifiedPrefix + `
@@ -1 +1 @@
-gord
+gourd
`[1:],
	}, {
		name: "delete_rune",
		in:   "groat\n",
		out:  "goat\n",
		unified: unifiedPrefix + `
@@ -1 +1 @@
-groat
+goat
`[1:],
	}, {
		name: "replace_rune",
		in:   "loud\n",
		out:  "lord\n",
		unified: unifiedPrefix + `
@@ -1 +1 @@
-loud
+lord
`[1:],
	}, {
		name: "insert_line",
		in:   "one\nthree\n",
		out:  "one\ntwo\nthree\n",
		unified: unifiedPrefix + `
@@ -1,2 +1,3 @@
 one
+two
 three
`[1:],
	}, {
		name: "replace_no_newline",
		in:   "A",
		out:  "B",
		unified: unifiedPrefix + `
@@ -1 +1 @@
-A
\ No newline at end of file
+B
\ No newline at end of file
`[1:],
	}, {
		name: "delete_front",
		in:   "A\nB\nC\nA\nB\nB\nA\n",
		out:  "C\nB\nA\nB\nA\nC\n",
		unified: unifiedPrefix + `
@@ -1,7 +1,6 @@
-A
-B
 C
+B
 A
 B
-B
 A
+C
`[1:],
		nodiff: true, // diff algorithm produces different delete/insert pattern
	},
		{
			name: "replace_last_line",
			in:   "A\nB\n",
			out:  "A\nC\n\n",
			unified: unifiedPrefix + `
@@ -1,2 +1,3 @@
 A
-B
+C
+
`[1:],
		},
		{
			name: "mulitple_replace",
			in:   "A\nB\nC\nD\nE\nF\nG\n",
			out:  "A\nH\nI\nJ\nE\nF\nK\n",
			unified: unifiedPrefix + `
@@ -1,7 +1,7 @@
 A
-B
-C
-D
+H
+I
+J
 E
 F
-G
+K
`[1:],
		}} {
		t.Run(test.name, func(t *testing.T) {
			t.Helper()
			edits := compute(span.FileURI("/"+test.name), test.in, test.out)
			got := diff.ApplyEdits(test.in, edits)
			unified := fmt.Sprint(diff.ToUnified("from", "to", test.in, edits))
			if got != test.out {
				t.Errorf("got patched:\n%v\nfrom diff:\n%v\nexpected:\n%v", got, unified, test.out)
			}
			if unified != test.unified {
				t.Errorf("got diff:\n%v\nexpected:\n%v", unified, test.unified)
			}
			if *verifyDiff && !test.nodiff {
				diff, err := getDiffOutput(test.in, test.out)
				if err != nil {
					t.Fatal(err)
				}
				if len(diff) > 0 {
					diff = unifiedPrefix + diff
				}
				if diff != test.unified {
					t.Errorf("unified:\n%q\ndiff -u:\n%q", test.unified, diff)
				}
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
