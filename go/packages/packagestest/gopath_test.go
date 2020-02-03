// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packagestest_test

import (
	"path/filepath"
	"testing"

	"golang.org/x/tools/go/packages/packagestest"
)

func TestGOPATHExport(t *testing.T) {
	exported := packagestest.Export(t, packagestest.GOPATH, testdata)
	defer exported.Cleanup()
	// Check that the cfg contains all the right bits
	var expectDir = filepath.Join(exported.Temp(), "fake1", "src")
	if exported.Config.Dir != expectDir {
		t.Errorf("Got working directory %v expected %v", exported.Config.Dir, expectDir)
	}
	checkFiles(t, exported, []fileTest{
		{"golang.org/fake1", "a.go", "fake1/src/golang.org/fake1/a.go", checkLink("testdata/a.go")},
		{"golang.org/fake1", "b.go", "fake1/src/golang.org/fake1/b.go", checkContent("package fake1")},
		{"golang.org/fake2", "other/a.go", "fake2/src/golang.org/fake2/other/a.go", checkContent("package fake2")},
		{"golang.org/fake2/v2", "other/a.go", "fake2_v2/src/golang.org/fake2/v2/other/a.go", checkContent("package fake2")},
	})
}

func TestGroupFilesByModules(t *testing.T) {
	for _, tt := range []struct {
		testdir string
		want    []packagestest.Module
	}{
		{
			testdir: "testdata/groups/one",
			want: []packagestest.Module{
				{
					Name: "testdata/groups/one",
					Files: map[string]interface{}{
						"main.go": true,
					},
				},
				{
					Name: "example.com/extra",
					Files: map[string]interface{}{
						"help.go": true,
					},
				},
			},
		},
		{
			testdir: "testdata/groups/two",
			want: []packagestest.Module{
				{
					Name: "testdata/groups/two",
					Files: map[string]interface{}{
						"main.go":      true,
						"expect/yo.go": true,
					},
				},
				{
					Name: "example.com/extra",
					Files: map[string]interface{}{
						"me.go":        true,
						"geez/help.go": true,
					},
				},
				{
					Name: "example.com/tempmod",
					Files: map[string]interface{}{
						"main.go": true,
					},
				},
			},
		},
	} {
		t.Run(tt.testdir, func(t *testing.T) {
			got, err := packagestest.GroupFilesByModules(tt.testdir)
			if err != nil {
				t.Fatalf("could not group files %v", err)
			}
			if len(got) != len(tt.want) {
				t.Fatalf("%s: wanted %d modules but got %d", tt.testdir, len(tt.want), len(got))
			}
			for i, w := range tt.want {
				g := got[i]
				if filepath.FromSlash(g.Name) != filepath.FromSlash(w.Name) {
					t.Fatalf("%s: wanted module[%d].Name to be %s but got %s", tt.testdir, i, filepath.FromSlash(w.Name), filepath.FromSlash(g.Name))
				}
				for fh := range w.Files {
					if _, ok := g.Files[fh]; !ok {
						t.Fatalf("%s, module[%d]: wanted %s but could not find", tt.testdir, i, fh)
					}
				}
				for fh := range g.Files {
					if _, ok := w.Files[fh]; !ok {
						t.Fatalf("%s, module[%d]: found unexpected file %s", tt.testdir, i, fh)
					}
				}
			}
		})
	}
}
