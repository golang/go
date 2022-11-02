// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modindex

import (
	"encoding/hex"
	"encoding/json"
	"go/build"
	"internal/diff"
	"path/filepath"
	"reflect"
	"runtime"
	"testing"
)

func init() {
	isTest = true
	enabled = true // to allow GODEBUG=goindex=0 go test, when things are very broken
}

func TestIndex(t *testing.T) {
	src := filepath.Join(runtime.GOROOT(), "src")
	checkPkg := func(t *testing.T, m *Module, pkg string, data []byte) {
		p := m.Package(pkg)
		bp, err := p.Import(build.Default, build.ImportComment)
		if err != nil {
			t.Fatal(err)
		}
		bp1, err := build.Default.Import(".", filepath.Join(src, pkg), build.ImportComment)
		if err != nil {
			t.Fatal(err)
		}

		if !reflect.DeepEqual(bp, bp1) {
			t.Errorf("mismatch")
			t.Logf("index:\n%s", hex.Dump(data))

			js, err := json.MarshalIndent(bp, "", "\t")
			if err != nil {
				t.Fatal(err)
			}
			js1, err := json.MarshalIndent(bp1, "", "\t")
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("diff:\n%s", diff.Diff("index", js, "correct", js1))
			t.FailNow()
		}
	}

	// Check packages in increasing complexity, one at a time.
	pkgs := []string{
		"crypto",
		"encoding",
		"unsafe",
		"encoding/json",
		"runtime",
		"net",
	}
	var raws []*rawPackage
	for _, pkg := range pkgs {
		raw := importRaw(src, pkg)
		raws = append(raws, raw)
		t.Run(pkg, func(t *testing.T) {
			data := encodeModuleBytes([]*rawPackage{raw})
			m, err := fromBytes(src, data)
			if err != nil {
				t.Fatal(err)
			}
			checkPkg(t, m, pkg, data)
		})
	}

	// Check that a multi-package index works too.
	t.Run("all", func(t *testing.T) {
		data := encodeModuleBytes(raws)
		m, err := fromBytes(src, data)
		if err != nil {
			t.Fatal(err)
		}
		for _, pkg := range pkgs {
			checkPkg(t, m, pkg, data)
		}
	})
}

func TestImportRaw_IgnoreNonGo(t *testing.T) {
	path := filepath.Join("testdata", "ignore_non_source")
	p := importRaw(path, ".")

	wantFiles := []string{"a.syso", "b.go", "c.c"}

	var gotFiles []string
	for i := range p.sourceFiles {
		gotFiles = append(gotFiles, p.sourceFiles[i].name)
	}

	if !reflect.DeepEqual(gotFiles, wantFiles) {
		t.Errorf("names of files in importRaw(testdata/ignore_non_source): got %v; want %v",
			gotFiles, wantFiles)
	}
}
