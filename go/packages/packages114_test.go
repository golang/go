// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.14

package packages_test

import (
	"fmt"
	"path/filepath"
	"testing"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/go/packages/packagestest"
)

// These tests check fixes that are only available in Go 1.14.
// They can be moved into packages_test.go when we no longer support 1.13.
// See golang/go#35973 for more information.
func TestInvalidFilesInOverlay(t *testing.T) { packagestest.TestAll(t, testInvalidFilesInOverlay) }
func testInvalidFilesInOverlay(t *testing.T, exporter packagestest.Exporter) {
	exported := packagestest.Export(t, exporter, []packagestest.Module{
		{
			Name: "golang.org/fake",
			Files: map[string]interface{}{
				"d/d.go":      `package d; import "net/http"; const d = http.MethodGet;`,
				"d/util.go":   ``,
				"d/d_test.go": ``,
			},
		},
	})
	defer exported.Cleanup()

	dir := filepath.Dir(filepath.Dir(exported.File("golang.org/fake", "d/d.go")))

	// Additional tests for test variants.
	for i, tt := range []struct {
		name    string
		overlay map[string][]byte
		want    string // expected value of d.D

	}{
		// Overlay with a test variant.
		{"test_variant",
			map[string][]byte{
				filepath.Join(dir, "d", "d_test.go"): []byte(`package d; import "testing"; const D = d + "_test"; func TestD(t *testing.T) {};`)},
			`"GET_test"`},
		// Overlay in package.
		{"second_file",
			map[string][]byte{
				filepath.Join(dir, "d", "util.go"): []byte(`package d; const D = d + "_util";`)},
			`"GET_util"`},
	} {
		t.Run(tt.name, func(t *testing.T) {
			exported.Config.Overlay = tt.overlay
			exported.Config.Mode = packages.NeedName | packages.NeedFiles | packages.NeedCompiledGoFiles |
				packages.NeedDeps | packages.NeedTypes | packages.NeedTypesSizes
			exported.Config.Tests = true

			for f := range tt.overlay {
				initial, err := packages.Load(exported.Config, fmt.Sprintf("file=%s", f))
				if err != nil {
					t.Fatal(err)
				}
				d := initial[0]
				var containsFile bool
				for _, goFile := range d.CompiledGoFiles {
					if f == goFile {
						containsFile = true
						break
					}
				}
				if !containsFile {
					t.Fatalf("expected %s in CompiledGoFiles, got %v", f, d.CompiledGoFiles)
				}
				// Check value of d.D.
				dD := constant(d, "D")
				if dD == nil {
					t.Fatalf("%d. d.D: got nil", i)
				}
				got := dD.Val().String()
				if got != tt.want {
					t.Fatalf("%d. d.D: got %s, want %s", i, got, tt.want)
				}
			}
		})
	}
}
