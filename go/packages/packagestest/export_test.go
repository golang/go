// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packagestest_test

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	"golang.org/x/tools/go/packages/packagestest"
)

var testdata = []packagestest.Module{{
	Name: "golang.org/fake1",
	Files: map[string]interface{}{
		"a.go": packagestest.Symlink("testdata/a.go"),
		"b.go": "package fake1",
	},
}, {
	Name: "golang.org/fake2",
	Files: map[string]interface{}{
		"other/a.go": "package fake2",
	},
}}

type fileTest struct {
	module, fragment, expect string
	check                    func(t *testing.T, filename string)
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
			test.check(t, got)
		}
	}
}

func checkLink(expect string) func(t *testing.T, filename string) {
	expect = filepath.FromSlash(expect)
	return func(t *testing.T, filename string) {
		if target, err := os.Readlink(filename); err != nil {
			t.Errorf("Error checking link %v: %v", filename, err)
		} else if target != expect {
			t.Errorf("Link %v does not match, got %v expected %v", filename, target, expect)
		}
	}
}

func checkContent(expect string) func(t *testing.T, filename string) {
	return func(t *testing.T, filename string) {
		if content, err := ioutil.ReadFile(filename); err != nil {
			t.Errorf("Error reading %v: %v", filename, err)
		} else if string(content) != expect {
			t.Errorf("Content of %v does not match, got %v expected %v", filename, string(content), expect)
		}
	}
}
