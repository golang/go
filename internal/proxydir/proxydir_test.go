// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proxydir

import (
	"archive/zip"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestWriteModuleVersion(t *testing.T) {
	tests := []struct {
		modulePath, version string
		files               map[string][]byte
	}{
		{
			modulePath: "mod.test/module",
			version:    "v1.2.3",
			files: map[string][]byte{
				"go.mod":   []byte("module mod.com\n\ngo 1.12"),
				"const.go": []byte("package module\n\nconst Answer = 42"),
			},
		},
		{
			modulePath: "mod.test/module",
			version:    "v1.2.4",
			files: map[string][]byte{
				"go.mod":   []byte("module mod.com\n\ngo 1.12"),
				"const.go": []byte("package module\n\nconst Answer = 43"),
			},
		},
		{
			modulePath: "mod.test/nogomod",
			version:    "v0.9.0",
			files: map[string][]byte{
				"const.go": []byte("package module\n\nconst Other = \"Other\""),
			},
		},
	}
	dir, err := os.MkdirTemp("", "proxydirtest-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)
	for _, test := range tests {
		// Since we later assert on the contents of /list, don't use subtests.
		if err := WriteModuleVersion(dir, test.modulePath, test.version, test.files); err != nil {
			t.Fatal(err)
		}
		rootDir := filepath.Join(dir, filepath.FromSlash(test.modulePath), "@v")
		gomod, err := os.ReadFile(filepath.Join(rootDir, test.version+".mod"))
		if err != nil {
			t.Fatal(err)
		}
		wantMod, ok := test.files["go.mod"]
		if !ok {
			wantMod = []byte("module " + test.modulePath)
		}
		if got, want := string(gomod), string(wantMod); got != want {
			t.Errorf("reading %s/@v/%s.mod: got %q, want %q", test.modulePath, test.version, got, want)
		}
		zr, err := zip.OpenReader(filepath.Join(rootDir, test.version+".zip"))
		if err != nil {
			t.Fatal(err)
		}
		defer zr.Close()

		for _, zf := range zr.File {
			r, err := zf.Open()
			if err != nil {
				t.Fatal(err)
			}
			defer r.Close()
			content, err := io.ReadAll(r)
			if err != nil {
				t.Fatal(err)
			}
			name := strings.TrimPrefix(zf.Name, fmt.Sprintf("%s@%s/", test.modulePath, test.version))
			if got, want := string(content), string(test.files[name]); got != want {
				t.Errorf("unzipping %q: got %q, want %q", zf.Name, got, want)
			}
			delete(test.files, name)
		}
		for name := range test.files {
			t.Errorf("file %q not present in the module zip", name)
		}
	}

	lists := []struct {
		modulePath, want string
	}{
		{"mod.test/module", "v1.2.3\nv1.2.4\n"},
		{"mod.test/nogomod", "v0.9.0\n"},
	}

	for _, test := range lists {
		fp := filepath.Join(dir, filepath.FromSlash(test.modulePath), "@v", "list")
		list, err := os.ReadFile(fp)
		if err != nil {
			t.Fatal(err)
		}
		if got := string(list); got != test.want {
			t.Errorf("%q/@v/list: got %q, want %q", test.modulePath, got, test.want)
		}
	}
}
