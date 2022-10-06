// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packagestest_test

import (
	"go/token"
	"testing"

	"golang.org/x/tools/go/expect"
	"golang.org/x/tools/go/packages/packagestest"
)

func TestExpect(t *testing.T) {
	exported := packagestest.Export(t, packagestest.GOPATH, []packagestest.Module{{
		Name:  "golang.org/fake",
		Files: packagestest.MustCopyFileTree("testdata"),
	}})
	defer exported.Cleanup()
	checkCount := 0
	if err := exported.Expect(map[string]interface{}{
		"check": func(src, target token.Position) {
			checkCount++
		},
		"boolArg": func(n *expect.Note, yes, no bool) {
			if !yes {
				t.Errorf("Expected boolArg first param to be true")
			}
			if no {
				t.Errorf("Expected boolArg second param to be false")
			}
		},
		"intArg": func(n *expect.Note, i int64) {
			if i != 42 {
				t.Errorf("Expected intarg to be 42")
			}
		},
		"stringArg": func(n *expect.Note, name expect.Identifier, value string) {
			if string(name) != value {
				t.Errorf("Got string arg %v expected %v", value, name)
			}
		},
		"directNote": func(n *expect.Note) {},
		"range": func(r packagestest.Range) {
			if r.Start == token.NoPos || r.Start == 0 {
				t.Errorf("Range had no valid starting position")
			}
			if r.End == token.NoPos || r.End == 0 {
				t.Errorf("Range had no valid ending position")
			} else if r.End <= r.Start {
				t.Errorf("Range ending was not greater than start")
			}
		},
		"checkEOF": func(n *expect.Note, p token.Pos) {
			if p <= n.Pos {
				t.Errorf("EOF was before the checkEOF note")
			}
		},
	}); err != nil {
		t.Fatal(err)
	}
	// We expect to have walked the @check annotations in all .go files,
	// including _test.go files (XTest or otherwise). But to have walked the
	// non-_test.go files only once. Hence wantCheck = 3 (testdata/test.go) + 1
	// (testdata/test_test.go) + 1 (testdata/x_test.go)
	wantCheck := 7
	if wantCheck != checkCount {
		t.Fatalf("Expected @check count of %v; got %v", wantCheck, checkCount)
	}
}
