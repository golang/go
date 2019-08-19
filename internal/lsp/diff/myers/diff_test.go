// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package myers_test

import (
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"reflect"
	"strings"
	"testing"

	"golang.org/x/tools/internal/lsp/diff/myers"
)

const (
	fileA         = "a/a.go"
	fileB         = "b/b.go"
	unifiedPrefix = "--- " + fileA + "\n+++ " + fileB + "\n"
)

var verifyDiff = flag.Bool("verify-diff", false, "Check that the unified diff output matches `diff -u`")

func TestDiff(t *testing.T) {
	for _, test := range []struct {
		a, b       string
		lines      []*myers.Op
		operations []*myers.Op
		unified    string
		nodiff     bool
	}{
		{
			a:          "A\nB\nC\n",
			b:          "A\nB\nC\n",
			operations: []*myers.Op{},
			unified: `
`[1:]}, {
			a: "A\n",
			b: "B\n",
			operations: []*myers.Op{
				&myers.Op{Kind: myers.Delete, I1: 0, I2: 1, J1: 0},
				&myers.Op{Kind: myers.Insert, Content: []string{"B\n"}, I1: 1, I2: 1, J1: 0},
			},
			unified: `
@@ -1 +1 @@
-A
+B
`[1:]}, {
			a: "A",
			b: "B",
			operations: []*myers.Op{
				&myers.Op{Kind: myers.Delete, I1: 0, I2: 1, J1: 0},
				&myers.Op{Kind: myers.Insert, Content: []string{"B"}, I1: 1, I2: 1, J1: 0},
			},
			unified: `
@@ -1 +1 @@
-A
\ No newline at end of file
+B
\ No newline at end of file
`[1:]}, {
			a: "A\nB\nC\nA\nB\nB\nA\n",
			b: "C\nB\nA\nB\nA\nC\n",
			operations: []*myers.Op{
				&myers.Op{Kind: myers.Delete, I1: 0, I2: 1, J1: 0},
				&myers.Op{Kind: myers.Delete, I1: 1, I2: 2, J1: 0},
				&myers.Op{Kind: myers.Insert, Content: []string{"B\n"}, I1: 3, I2: 3, J1: 1},
				&myers.Op{Kind: myers.Delete, I1: 5, I2: 6, J1: 4},
				&myers.Op{Kind: myers.Insert, Content: []string{"C\n"}, I1: 7, I2: 7, J1: 5},
			},
			unified: `
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
			a: "A\nB\n",
			b: "A\nC\n\n",
			operations: []*myers.Op{
				&myers.Op{Kind: myers.Delete, I1: 1, I2: 2, J1: 1},
				&myers.Op{Kind: myers.Insert, Content: []string{"C\n"}, I1: 2, I2: 2, J1: 1},
				&myers.Op{Kind: myers.Insert, Content: []string{"\n"}, I1: 2, I2: 2, J1: 2},
			},
			unified: `
@@ -1,2 +1,3 @@
 A
-B
+C
+
`[1:],
		},
		{
			a: "A\nB\nC\nD\nE\nF\nG\n",
			b: "A\nH\nI\nJ\nE\nF\nK\n",
			unified: `
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
`[1:]},
	} {
		a := myers.SplitLines(test.a)
		b := myers.SplitLines(test.b)
		ops := myers.Operations(a, b)
		if test.operations != nil {
			if len(ops) != len(test.operations) {
				t.Fatalf("expected %v operations, got %v", len(test.operations), len(ops))
			}
			for i, got := range ops {
				want := test.operations[i]
				if !reflect.DeepEqual(want, got) {
					t.Errorf("expected %v, got %v", want, got)
				}
			}
		}
		applied := myers.ApplyEdits(a, ops)
		for i, want := range applied {
			got := b[i]
			if got != want {
				t.Errorf("expected %v got %v", want, got)
			}
		}
		if test.unified != "" {
			diff := myers.ToUnified(fileA, fileB, a, ops)
			got := fmt.Sprint(diff)
			if !strings.HasPrefix(got, unifiedPrefix) {
				t.Errorf("expected prefix:\n%s\ngot:\n%s", unifiedPrefix, got)
				continue
			}
			got = got[len(unifiedPrefix):]
			if test.unified != got {
				t.Errorf("expected:\n%q\ngot:\n%q", test.unified, got)
			}
		}
		if *verifyDiff && test.unified != "" && !test.nodiff {
			diff, err := getDiffOutput(test.a, test.b)
			if err != nil {
				t.Fatal(err)
			}
			if diff != test.unified {
				t.Errorf("unified:\n%q\ndiff -u:\n%q", test.unified, diff)
			}
		}
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
	bits := strings.SplitN(diff, "\n", 3)
	if len(bits) != 3 {
		return "", fmt.Errorf("diff output did not have file prefix:\n%s", diff)
	}
	return bits[2], nil
}
