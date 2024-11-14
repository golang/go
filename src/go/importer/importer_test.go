// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package importer

import (
	"go/build"
	"go/token"
	"internal/testenv"
	"io"
	"os"
	"strings"
	"testing"
)

func TestMain(m *testing.M) {
	build.Default.GOROOT = testenv.GOROOT(nil)
	os.Exit(m.Run())
}

func TestForCompiler(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	const thePackage = "math/big"
	out, err := testenv.Command(t, testenv.GoToolPath(t), "list", "-export", "-f={{context.Compiler}}:{{.Export}}", thePackage).CombinedOutput()
	if err != nil {
		t.Fatalf("go list %s: %v\n%s", thePackage, err, out)
	}
	export := strings.TrimSpace(string(out))
	compiler, target, _ := strings.Cut(export, ":")

	if compiler == "gccgo" {
		t.Skip("golang.org/issue/22500")
	}

	fset := token.NewFileSet()

	t.Run("LookupDefault", func { t ->
		imp := ForCompiler(fset, compiler, nil)
		pkg, err := imp.Import(thePackage)
		if err != nil {
			t.Fatal(err)
		}
		if pkg.Path() != thePackage {
			t.Fatalf("Path() = %q, want %q", pkg.Path(), thePackage)
		}

		// Check that the fileset positions are accurate.
		// https://github.com/golang/go#28995
		mathBigInt := pkg.Scope().Lookup("Int")
		posn := fset.Position(mathBigInt.Pos()) // "$GOROOT/src/math/big/int.go:25:1"
		filename := strings.Replace(posn.Filename, "$GOROOT", testenv.GOROOT(t), 1)
		data, err := os.ReadFile(filename)
		if err != nil {
			t.Fatalf("can't read file containing declaration of math/big.Int: %v", err)
		}
		lines := strings.Split(string(data), "\n")
		if posn.Line > len(lines) || !strings.HasPrefix(lines[posn.Line-1], "type Int") {
			t.Fatalf("Object %v position %s does not contain its declaration",
				mathBigInt, posn)
		}
	})

	t.Run("LookupCustom", func { t ->
		// TODO(mdempsky): Decide whether to remove this test, or to fix
		// support for it in unified IR. It's not clear that we actually
		// need to support importing "math/big" as "math/bigger", for
		// example. cmd/link no longer supports that.
		if true /* was buildcfg.Experiment.Unified */{
			t.Skip("not supported by GOEXPERIMENT=unified; see go.dev/cl/406319")
		}

		lookup := func(path string) (io.ReadCloser, error) {
			if path != "math/bigger" {
				t.Fatalf("lookup called with unexpected path %q", path)
			}
			f, err := os.Open(target)
			if err != nil {
				t.Fatal(err)
			}
			return f, nil
		}
		imp := ForCompiler(fset, compiler, lookup)
		pkg, err := imp.Import("math/bigger")
		if err != nil {
			t.Fatal(err)
		}
		// Even though we open math/big.a, the import request was for math/bigger
		// and that should be recorded in pkg.Path(), at least for the gc toolchain.
		if pkg.Path() != "math/bigger" {
			t.Fatalf("Path() = %q, want %q", pkg.Path(), "math/bigger")
		}
	})
}
