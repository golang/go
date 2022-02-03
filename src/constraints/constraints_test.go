// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package constraints

import (
	"bytes"
	"fmt"
	"internal/testenv"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

type (
	testSigned[T Signed]     struct{ f T }
	testUnsigned[T Unsigned] struct{ f T }
	testInteger[T Integer]   struct{ f T }
	testFloat[T Float]       struct{ f T }
	testComplex[T Complex]   struct{ f T }
	testOrdered[T Ordered]   struct{ f T }
)

// TestTypes passes if it compiles.
type TestTypes struct {
	_ testSigned[int]
	_ testSigned[int64]
	_ testUnsigned[uint]
	_ testUnsigned[uintptr]
	_ testInteger[int8]
	_ testInteger[uint8]
	_ testInteger[uintptr]
	_ testFloat[float32]
	_ testComplex[complex64]
	_ testOrdered[int]
	_ testOrdered[float64]
	_ testOrdered[string]
}

var prolog = []byte(`
package constrainttest

import "constraints"

type (
	testSigned[T constraints.Signed]     struct{ f T }
	testUnsigned[T constraints.Unsigned] struct{ f T }
	testInteger[T constraints.Integer]   struct{ f T }
	testFloat[T constraints.Float]       struct{ f T }
	testComplex[T constraints.Complex]   struct{ f T }
	testOrdered[T constraints.Ordered]   struct{ f T }
)
`)

func TestFailure(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	gocmd := testenv.GoToolPath(t)
	tmpdir := t.TempDir()

	if err := os.WriteFile(filepath.Join(tmpdir, "go.mod"), []byte("module constraintest"), 0666); err != nil {
		t.Fatal(err)
	}

	// Test for types that should not satisfy a constraint.
	// For each pair of constraint and type, write a Go file
	//     var V constraint[type]
	// For example,
	//     var V testSigned[uint]
	// This should not compile, as testSigned (above) uses
	// constraints.Signed, and uint does not satisfy that constraint.
	// Therefore, the build of that code should fail.
	for i, test := range []struct {
		constraint, typ string
	}{
		{"testSigned", "uint"},
		{"testUnsigned", "int"},
		{"testInteger", "float32"},
		{"testFloat", "int8"},
		{"testComplex", "float64"},
		{"testOrdered", "bool"},
	} {
		i := i
		test := test
		t.Run(fmt.Sprintf("%s %d", test.constraint, i), func(t *testing.T) {
			t.Parallel()
			name := fmt.Sprintf("go%d.go", i)
			f, err := os.Create(filepath.Join(tmpdir, name))
			if err != nil {
				t.Fatal(err)
			}
			if _, err := f.Write(prolog); err != nil {
				t.Fatal(err)
			}
			if _, err := fmt.Fprintf(f, "var V %s[%s]\n", test.constraint, test.typ); err != nil {
				t.Fatal(err)
			}
			if err := f.Close(); err != nil {
				t.Fatal(err)
			}
			cmd := exec.Command(gocmd, "build", name)
			cmd.Dir = tmpdir
			if out, err := cmd.CombinedOutput(); err == nil {
				t.Error("build succeeded, but expected to fail")
			} else if len(out) > 0 {
				t.Logf("%s", out)
				const want = "does not implement"
				if !bytes.Contains(out, []byte(want)) {
					t.Errorf("output does not include %q", want)
				}
			} else {
				t.Error("no error output, expected something")
			}
		})
	}
}
