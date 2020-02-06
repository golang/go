// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packagestest_test

import (
	"os"
	"path/filepath"
	"testing"

	"golang.org/x/tools/go/packages/packagestest"
)

var testdata = []packagestest.Module{{
	Name: "golang.org/fake1",
	Files: map[string]interface{}{
		"a.go": packagestest.Symlink("testdata/a.go"),
		"b.go": "invalid file contents",
	},
	Overlay: map[string][]byte{
		"b.go": []byte("package fake1"),
		"c.go": []byte("package fake1"),
	},
}, {
	Name: "golang.org/fake2",
	Files: map[string]interface{}{
		"other/a.go": "package fake2",
	},
}, {
	Name: "golang.org/fake2/v2",
	Files: map[string]interface{}{
		"other/a.go": "package fake2",
	},
}, {
	Name: "golang.org/fake3@v1.0.0",
	Files: map[string]interface{}{
		"other/a.go": "package fake3",
	},
}, {
	Name: "golang.org/fake3@v1.1.0",
	Files: map[string]interface{}{
		"other/a.go": "package fake3",
	},
}}

type fileTest struct {
	module, fragment, expect string
	check                    func(t *testing.T, exported *packagestest.Exported, filename string)
}

func checkFiles(t *testing.T, exported *packagestest.Exported, tests []fileTest) {
	for _, test := range tests {
		expect := filepath.Join(exported.Temp(), filepath.FromSlash(test.expect))
		got := exported.File(test.module, test.fragment)
		if got == "" {
			t.Errorf("File %v missing from the output", expect)
		} else if got != expect {
			t.Errorf("Got file %v, expected %v", got, expect)
		}
		if test.check != nil {
			test.check(t, exported, got)
		}
	}
}

func checkLink(expect string) func(t *testing.T, exported *packagestest.Exported, filename string) {
	expect = filepath.FromSlash(expect)
	return func(t *testing.T, exported *packagestest.Exported, filename string) {
		if target, err := os.Readlink(filename); err != nil {
			t.Errorf("Error checking link %v: %v", filename, err)
		} else if target != expect {
			t.Errorf("Link %v does not match, got %v expected %v", filename, target, expect)
		}
	}
}

func checkContent(expect string) func(t *testing.T, exported *packagestest.Exported, filename string) {
	return func(t *testing.T, exported *packagestest.Exported, filename string) {
		if content, err := exported.FileContents(filename); err != nil {
			t.Errorf("Error reading %v: %v", filename, err)
		} else if string(content) != expect {
			t.Errorf("Content of %v does not match, got %v expected %v", filename, string(content), expect)
		}
	}
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
						"yo.go":        true,
						"geez/help.go": true,
					},
				},
				{
					Name: "example.com/extra/v2",
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
				{
					Name: "example.com/what@v1.0.0",
					Files: map[string]interface{}{
						"main.go": true,
					},
				},
				{
					Name: "example.com/what@v1.1.0",
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
