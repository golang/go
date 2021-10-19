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
	testSigned[T Signed]                      struct{ f T }
	testUnsigned[T Unsigned]                  struct{ f T }
	testInteger[T Integer]                    struct{ f T }
	testFloat[T Float]                        struct{ f T }
	testComplex[T Complex]                    struct{ f T }
	testOrdered[T Ordered]                    struct{ f T }
	testSlice[T Slice[E], E any]              struct{ f T }
	testMap[T Map[K, V], K comparable, V any] struct{ f T }
	testChan[T Chan[E], E any]                struct{ f T }
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
	_ testSlice[[]int, int]
	_ testMap[map[int]bool, int, bool]
	_ testChan[chan int, int]
}

func infer1[S Slice[E], E any](s S, v E) S                     { return s }
func infer2[M Map[K, V], K comparable, V any](m M, k K, v V) M { return m }
func infer3[C Chan[E], E any](c C, v E) C                      { return c }

func TestInference(t *testing.T) {
	var empty interface{}

	type S []int
	empty = infer1(S{}, 0)
	if _, ok := empty.(S); !ok {
		t.Errorf("infer1(S) returned %T, expected S", empty)
	}

	type M map[int]bool
	empty = infer2(M{}, 0, false)
	if _, ok := empty.(M); !ok {
		t.Errorf("infer2(M) returned %T, expected M", empty)
	}

	type C chan bool
	empty = infer3(make(C), true)
	if _, ok := empty.(C); !ok {
		t.Errorf("infer3(C) returned %T, expected C", empty)
	}
}

var prolog = []byte(`
package constrainttest

import "constraints"

type (
	testSigned[T constraints.Signed]                      struct{ f T }
	testUnsigned[T constraints.Unsigned]                  struct{ f T }
	testInteger[T constraints.Integer]                    struct{ f T }
	testFloat[T constraints.Float]                        struct{ f T }
	testComplex[T constraints.Complex]                    struct{ f T }
	testOrdered[T constraints.Ordered]                    struct{ f T }
	testSlice[T constraints.Slice[E], E any]              struct{ f T }
	testMap[T constraints.Map[K, V], K comparable, V any] struct{ f T }
	testChan[T constraints.Chan[E], E any]                struct{ f T }
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
		{"testSlice", "int, int"},
		{"testMap", "string, string, string"},
		{"testChan", "[]int, int"},
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
				const want = "does not satisfy"
				if !bytes.Contains(out, []byte(want)) {
					t.Errorf("output does not include %q", want)
				}
			} else {
				t.Error("no error output, expected something")
			}
		})
	}
}
