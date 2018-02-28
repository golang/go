// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package importer

import (
	"internal/testenv"
	"io"
	"os"
	"os/exec"
	"strings"
	"testing"
)

func TestFor(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	const thePackage = "math/big"
	out, err := exec.Command(testenv.GoToolPath(t), "list", "-f={{context.Compiler}}:{{.Target}}", thePackage).CombinedOutput()
	if err != nil {
		t.Fatalf("go list %s: %v\n%s", thePackage, err, out)
	}
	target := strings.TrimSpace(string(out))
	i := strings.Index(target, ":")
	compiler, target := target[:i], target[i+1:]
	if !strings.HasSuffix(target, ".a") {
		t.Fatalf("unexpected package %s target %q (not *.a)", thePackage, target)
	}

	if compiler == "gccgo" {
		t.Skip("golang.org/issue/22500")
	}

	t.Run("LookupDefault", func(t *testing.T) {
		imp := For(compiler, nil)
		pkg, err := imp.Import(thePackage)
		if err != nil {
			t.Fatal(err)
		}
		if pkg.Path() != thePackage {
			t.Fatalf("Path() = %q, want %q", pkg.Path(), thePackage)
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
		imp := For(compiler, lookup)
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
