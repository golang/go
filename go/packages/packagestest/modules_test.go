// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.11

package packagestest_test

import (
	"path/filepath"
	"testing"

	"golang.org/x/tools/go/packages/packagestest"
)

func TestModulesExport(t *testing.T) {
	exported := packagestest.Export(t, packagestest.Modules, testdata)
	defer exported.Cleanup()
	// Check that the cfg contains all the right bits
	var expectDir = filepath.Join(exported.Temp(), "primarymod/fake1")
	if exported.Config.Dir != expectDir {
		t.Errorf("Got working directory %v expected %v", exported.Config.Dir, expectDir)
	}
	checkFiles(t, exported, []fileTest{
		{"golang.org/fake1", "go.mod", "primarymod/fake1/go.mod", nil},
		{"golang.org/fake1", "a.go", "primarymod/fake1/a.go", checkLink("testdata/a.go")},
		{"golang.org/fake1", "b.go", "primarymod/fake1/b.go", checkContent("package fake1")},
		{"golang.org/fake2", "go.mod", "modcache/pkg/mod/golang.org/fake2@v1.0.0/go.mod", nil},
		{"golang.org/fake2", "other/a.go", "modcache/pkg/mod/golang.org/fake2@v1.0.0/other/a.go", checkContent("package fake2")},
	})
}
