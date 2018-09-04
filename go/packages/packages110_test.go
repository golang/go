// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !go1.11

package packages_test

import (
	"bytes"
	"fmt"
	"os"
	"strings"
	"testing"

	"golang.org/x/tools/go/packages"
)

func init() {
	usesOldGolist = true
}

func TestXTestImports(t *testing.T) {
	tmp, cleanup := makeTree(t, map[string]string{
		"src/a/a_test.go": `package a_test; import "a"`,
		"src/a/a.go":      `package a`,
	})
	defer cleanup()

	cfg := &packages.Config{
		Mode:  packages.LoadImports,
		Dir:   tmp,
		Env:   append(os.Environ(), "GOPATH="+tmp, "GO111MODULE=off"),
		Tests: true,
	}
	initial, err := packages.Load(cfg, "a")
	if err != nil {
		t.Fatal(err)
	}

	var gotImports bytes.Buffer
	for _, pkg := range initial {
		var imports []string
		for imp, pkg := range pkg.Imports {
			imports = append(imports, fmt.Sprintf("%q: %q", imp, pkg.ID))
		}
		fmt.Fprintf(&gotImports, "%s {%s}\n", pkg.ID, strings.Join(imports, ", "))
	}
	wantImports := `a {}
a [a.test] {}
a_test [a.test] {"a": "a [a.test]"}
`
	if gotImports.String() != wantImports {
		t.Fatalf("wrong imports: got <<%s>>, want <<%s>>", gotImports.String(), wantImports)
	}
}
