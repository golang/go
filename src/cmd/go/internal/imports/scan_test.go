// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package imports

import (
	"bytes"
	"internal/testenv"
	"io/ioutil"
	"path"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestScan(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	imports, testImports, err := ScanDir(filepath.Join(runtime.GOROOT(), "src/encoding/json"), Tags())
	if err != nil {
		t.Fatal(err)
	}
	foundBase64 := false
	for _, p := range imports {
		if p == "encoding/base64" {
			foundBase64 = true
		}
		if p == "encoding/binary" {
			// A dependency but not an import
			t.Errorf("json reported as importing encoding/binary but does not")
		}
		if p == "net/http" {
			// A test import but not an import
			t.Errorf("json reported as importing encoding/binary but does not")
		}
	}
	if !foundBase64 {
		t.Errorf("json missing import encoding/base64 (%q)", imports)
	}

	foundHTTP := false
	for _, p := range testImports {
		if p == "net/http" {
			foundHTTP = true
		}
		if p == "unicode/utf16" {
			// A package import but not a test import
			t.Errorf("json reported as test-importing unicode/utf16  but does not")
		}
	}
	if !foundHTTP {
		t.Errorf("json missing test import net/http (%q)", testImports)
	}
}
func TestScanDir(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	dirs, err := ioutil.ReadDir("testdata")
	if err != nil {
		t.Fatal(err)
	}
	for _, dir := range dirs {
		if !dir.IsDir() || strings.HasPrefix(dir.Name(), ".") {
			continue
		}
		t.Run(dir.Name(), func(t *testing.T) {
			tagsData, err := ioutil.ReadFile(filepath.Join("testdata", dir.Name(), "tags.txt"))
			if err != nil {
				t.Fatalf("error reading tags: %v", err)
			}
			tags := make(map[string]bool)
			for _, t := range strings.Fields(string(tagsData)) {
				tags[t] = true
			}

			wantData, err := ioutil.ReadFile(filepath.Join("testdata", dir.Name(), "want.txt"))
			if err != nil {
				t.Fatalf("error reading want: %v", err)
			}
			want := string(bytes.TrimSpace(wantData))

			imports, _, err := ScanDir(path.Join("testdata", dir.Name()), tags)
			if err != nil {
				t.Fatal(err)
			}
			got := strings.Join(imports, "\n")
			if got != want {
				t.Errorf("ScanDir: got imports:\n%s\n\nwant:\n%s", got, want)
			}
		})
	}
}
