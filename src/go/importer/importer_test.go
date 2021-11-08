// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package importer

import (
	"go/token"
	"internal/testenv"
	"io"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"testing"
)

func TestForCompiler(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	const thePackage = "math/big"
	out, err := exec.Command(testenv.GoToolPath(t), "list", "-f={{context.Compiler}}:{{.Target}}", thePackage).CombinedOutput()
	if err != nil {
		t.Fatalf("go list %s: %v\n%s", thePackage, err, out)
	}
	target := strings.TrimSpace(string(out))
	compiler, target, _ := strings.Cut(target, ":")
	if !strings.HasSuffix(target, ".a") {
		t.Fatalf("unexpected package %s target %q (not *.a)", thePackage, target)
	}

	if compiler == "gccgo" {
		t.Skip("golang.org/issue/22500")
	}

	fset := token.NewFileSet()

	t.Run("LookupDefault", func(t *testing.T) {
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
		filename := strings.Replace(posn.Filename, "$GOROOT", runtime.GOROOT(), 1)
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

	t.Run("LookupCustom", func(t *testing.T) {
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
