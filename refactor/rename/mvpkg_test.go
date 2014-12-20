// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// licence that can be found in the LICENSE file.

package rename

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/build"
	"go/format"
	"go/token"
	"io/ioutil"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"golang.org/x/tools/go/buildutil"
)

func TestErrors(t *testing.T) {
	tests := []struct {
		ctxt     *build.Context
		from, to string
		want     string // regexp to match error, or "OK"
	}{
		// Simple example.
		{
			ctxt: fakeContext(map[string][]string{
				"foo": {`package foo; type T int`},
				"bar": {`package bar`},
				"main": {`package main

import "foo"

var _ foo.T
`},
			}),
			from: "foo", to: "bar",
			want: "invalid move destination: bar conflicts with directory /go/src/bar",
		},
		// Subpackage already exists.
		{
			ctxt: fakeContext(map[string][]string{
				"foo":     {`package foo; type T int`},
				"foo/sub": {`package sub`},
				"bar/sub": {`package sub`},
				"main": {`package main

import "foo"

var _ foo.T
`},
			}),
			from: "foo", to: "bar",
			want: "invalid move destination: bar; package or subpackage bar/sub already exists",
		},
		// Invalid base name.
		{
			ctxt: fakeContext(map[string][]string{
				"foo": {`package foo; type T int`},
				"main": {`package main

import "foo"

var _ foo.T
`},
			}),
			from: "foo", to: "bar-v2.0",
			want: "invalid move destination: bar-v2.0; gomvpkg does not " +
				"support move destinations whose base names are not valid " +
				"go identifiers",
		},
	}

	for _, test := range tests {
		ctxt := test.ctxt

		got := make(map[string]string)
		rewriteFile = func(fset *token.FileSet, f *ast.File, orig string) error {
			var out bytes.Buffer
			if err := format.Node(&out, fset, f); err != nil {
				return err
			}
			got[orig] = out.String()
			return nil
		}
		moveDirectory = func(from, to string) error {
			for path, contents := range got {
				if strings.HasPrefix(path, from) {
					newPath := strings.Replace(path, from, to, 1)
					delete(got, path)
					got[newPath] = contents
				}
			}
			return nil
		}

		err := Move(ctxt, test.from, test.to, "")
		prefix := fmt.Sprintf("-from %q -to %q", test.from, test.to)
		if err == nil {
			t.Errorf("%s: nil error. Expected error: %s", prefix, test.want)
			continue
		}
		if test.want != err.Error() {
			t.Errorf("%s: conflict does not match expectation:\n"+
				"Error: %q\n"+
				"Pattern: %q",
				prefix, err.Error(), test.want)
		}
	}
}

func TestMoves(t *testing.T) {
	tests := []struct {
		ctxt     *build.Context
		from, to string
		want     map[string]string
	}{
		// Simple example.
		{
			ctxt: fakeContext(map[string][]string{
				"foo": {`package foo; type T int`},
				"main": {`package main

import "foo"

var _ foo.T
`},
			}),
			from: "foo", to: "bar",
			want: map[string]string{
				"/go/src/main/0.go": `package main

import "bar"

var _ bar.T
`,
				"/go/src/bar/0.go": `package bar

type T int
`,
			},
		},

		// Example with subpackage.
		{
			ctxt: fakeContext(map[string][]string{
				"foo":     {`package foo; type T int`},
				"foo/sub": {`package sub; type T int`},
				"main": {`package main

import "foo"
import "foo/sub"

var _ foo.T
var _ sub.T
`},
			}),
			from: "foo", to: "bar",
			want: map[string]string{
				"/go/src/main/0.go": `package main

import "bar"
import "bar/sub"

var _ bar.T
var _ sub.T
`,
				"/go/src/bar/0.go": `package bar

type T int
`,
				"/go/src/bar/sub/0.go": `package sub; type T int`,
			},
		},
	}

	for _, test := range tests {
		ctxt := test.ctxt

		var mu sync.Mutex
		got := make(map[string]string)
		// Populate got with starting file set. rewriteFile and moveDirectory
		// will mutate got to produce resulting file set.
		buildutil.ForEachPackage(ctxt, func(importPath string, err error) {
			if err != nil {
				return
			}
			path := filepath.Join("/go/src", importPath, "0.go")
			if !buildutil.FileExists(ctxt, path) {
				return
			}
			f, err := ctxt.OpenFile(path)
			defer f.Close()
			if err != nil {
				t.Errorf("unexpected error opening file: %s", err)
				return
			}
			bytes, err := ioutil.ReadAll(f)
			if err != nil {
				t.Errorf("unexpected error reading file: %s", err)
				return
			}
			mu.Lock()
			got[path] = string(bytes)
			defer mu.Unlock()
		})
		rewriteFile = func(fset *token.FileSet, f *ast.File, orig string) error {
			var out bytes.Buffer
			if err := format.Node(&out, fset, f); err != nil {
				return err
			}
			got[orig] = out.String()
			return nil
		}
		moveDirectory = func(from, to string) error {
			for path, contents := range got {
				if strings.HasPrefix(path, from) {
					newPath := strings.Replace(path, from, to, 1)
					delete(got, path)
					got[newPath] = contents
				}
			}
			return nil
		}

		err := Move(ctxt, test.from, test.to, "")
		prefix := fmt.Sprintf("-from %q -to %q", test.from, test.to)
		if err != nil {
			t.Errorf("%s: unexpected error: %s", prefix, err)
			continue
		}

		for file, wantContent := range test.want {
			gotContent, ok := got[file]
			delete(got, file)
			if !ok {
				// TODO(matloob): some testcases might have files that won't be
				// rewritten
				t.Errorf("%s: file %s not rewritten", prefix, file)
				continue
			}
			if gotContent != wantContent {
				t.Errorf("%s: rewritten file %s does not match expectation; got <<<%s>>>\n"+
					"want <<<%s>>>", prefix, file, gotContent, wantContent)
			}
		}
		// got should now be empty
		for file := range got {
			t.Errorf("%s: unexpected rewrite of file %s", prefix, file)
		}
	}
}
