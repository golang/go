// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package imports

import (
	"internal/testenv"
	"path/filepath"
	"reflect"
	"runtime"
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

func TestScanStar(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	imports, _, err := ScanDir("testdata/import1", map[string]bool{"*": true})
	if err != nil {
		t.Fatal(err)
	}

	want := []string{"import1", "import2", "import3", "import4"}
	if !reflect.DeepEqual(imports, want) {
		t.Errorf("ScanDir testdata/import1:\nhave %v\nwant %v", imports, want)
	}
}
