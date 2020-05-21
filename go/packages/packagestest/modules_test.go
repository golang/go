// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packagestest_test

import (
	"path/filepath"
	"testing"

	"golang.org/x/tools/go/packages/packagestest"
	"golang.org/x/tools/internal/testenv"
)

func TestModulesExport(t *testing.T) {
	testenv.NeedsGo1Point(t, 11)
	exported := packagestest.Export(t, packagestest.Modules, testdata)
	defer exported.Cleanup()
	// Check that the cfg contains all the right bits
	var expectDir = filepath.Join(exported.Temp(), "fake1")
	if exported.Config.Dir != expectDir {
		t.Errorf("Got working directory %v expected %v", exported.Config.Dir, expectDir)
	}
	checkFiles(t, exported, []fileTest{
		{"golang.org/fake1", "go.mod", "fake1/go.mod", nil},
		{"golang.org/fake1", "a.go", "fake1/a.go", checkLink("testdata/a.go")},
		{"golang.org/fake1", "b.go", "fake1/b.go", checkContent("package fake1")},
		{"golang.org/fake2", "go.mod", "modcache/pkg/mod/golang.org/fake2@v1.0.0/go.mod", nil},
		{"golang.org/fake2", "other/a.go", "modcache/pkg/mod/golang.org/fake2@v1.0.0/other/a.go", checkContent("package fake2")},
		{"golang.org/fake2/v2", "other/a.go", "modcache/pkg/mod/golang.org/fake2/v2@v2.0.0/other/a.go", checkContent("package fake2")},
		{"golang.org/fake3@v1.1.0", "other/a.go", "modcache/pkg/mod/golang.org/fake3@v1.1.0/other/a.go", checkContent("package fake3")},
		{"golang.org/fake3@v1.0.0", "other/a.go", "modcache/pkg/mod/golang.org/fake3@v1.0.0/other/a.go", nil},
	})
}
