// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tokeninternal_test

import (
	"fmt"
	"go/token"
	"strings"
	"testing"

	"golang.org/x/tools/internal/tokeninternal"
)

func TestAddExistingFiles(t *testing.T) {
	fset := token.NewFileSet()

	check := func(descr, want string) {
		t.Helper()
		if got := fsetString(fset); got != want {
			t.Errorf("%s: got %s, want %s", descr, got, want)
		}
	}

	fileA := fset.AddFile("A", -1, 3)
	fileB := fset.AddFile("B", -1, 5)
	_ = fileB
	check("after AddFile [AB]", "{A:1-4 B:5-10}")

	tokeninternal.AddExistingFiles(fset, nil)
	check("after AddExistingFiles []", "{A:1-4 B:5-10}")

	fileC := token.NewFileSet().AddFile("C", 100, 5)
	fileD := token.NewFileSet().AddFile("D", 200, 5)
	tokeninternal.AddExistingFiles(fset, []*token.File{fileC, fileA, fileD, fileC})
	check("after AddExistingFiles [CADC]", "{A:1-4 B:5-10 C:100-105 D:200-205}")

	fileE := fset.AddFile("E", -1, 3)
	_ = fileE
	check("after AddFile [E]", "{A:1-4 B:5-10 C:100-105 D:200-205 E:206-209}")
}

func fsetString(fset *token.FileSet) string {
	var buf strings.Builder
	buf.WriteRune('{')
	sep := ""
	fset.Iterate(func(f *token.File) bool {
		fmt.Fprintf(&buf, "%s%s:%d-%d", sep, f.Name(), f.Base(), f.Base()+f.Size())
		sep = " "
		return true
	})
	buf.WriteRune('}')
	return buf.String()
}
