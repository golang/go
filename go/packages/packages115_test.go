// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.15

package packages_test

import (
	"testing"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/go/packages/packagestest"
)

// TestInvalidFilesInXTest checks the fix for golang/go#37971.
func TestInvalidFilesInXTest(t *testing.T) {
	packagestest.TestAll(t, testInvalidFilesInXTest)
}
func testInvalidFilesInXTest(t *testing.T, exporter packagestest.Exporter) {
	exported := packagestest.Export(t, exporter, []packagestest.Module{
		{
			Name: "golang.org/fake",
			Files: map[string]interface{}{
				"d/d.go":      `package d; import "net/http"; const d = http.MethodGet; func Get() string { return d; }`,
				"d/d2.go":     ``, // invalid file
				"d/d_test.go": `package d_test; import "testing"; import "golang.org/fake/d"; func TestD(t *testing.T) { d.Get(); }`,
			},
		},
	})
	defer exported.Cleanup()

	exported.Config.Mode = packages.NeedName | packages.NeedFiles
	exported.Config.Tests = true

	initial, err := packages.Load(exported.Config, "golang.org/fake/d")
	if err != nil {
		t.Fatal(err)
	}
	if len(initial) != 3 {
		t.Errorf("expected 3 packages, got %d", len(initial))
	}
}
