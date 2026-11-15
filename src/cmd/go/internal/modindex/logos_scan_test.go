// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modindex

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"cmd/go/internal/cfg"
	"cmd/go/internal/imports"
)

func setLogosMode(t *testing.T) { applyLogosMode(t, true) }

func setPlainGoMode(t *testing.T) { applyLogosMode(t, false) }

func applyLogosMode(t *testing.T, logos bool) {
	t.Helper()
	old := cfg.Logos
	cfg.Logos = logos
	t.Cleanup(func() { cfg.Logos = old })
}

func indexPackageForDir(t *testing.T, modroot, reldir string) *IndexPackage {
	t.Helper()
	raw := importRaw(modroot, reldir)
	if raw.error != "" {
		t.Fatalf("importRaw(%s, %s): %s", modroot, reldir, raw.error)
	}
	pkg, err := packageFromBytes(modroot, encodePackageBytes(raw))
	if err != nil {
		t.Fatalf("packageFromBytes: %v", err)
	}
	return pkg
}

func TestScanDirLogosIndexed(t *testing.T) {
	modroot := t.TempDir()
	pkgdir := filepath.Join(modroot, "p")
	if err := os.MkdirAll(pkgdir, 0o777); err != nil {
		t.Fatal(err)
	}
	files := []struct{ name, src string }{
		{"a.go", "package p\n\nimport _ \"goimport\"\n"},
		{"a.lgo", "package p\n\nimport _ \"logosimport\"\n"},
		{"a_test.lgo", "package p_test\n\nimport _ \"logostestimport\"\n"},
		{"alias.\u26a1", "package p\n\nimport _ \"logosaliasimport\"\n"},
		{"scroll.\U0001f4dc", "package p\n\nimport _ \"logosscrollimport\"\n"},
		{"gopher.\u0295\u25d4\u03d6\u25d4\u0294", "package p\n\nimport _ \"logosgopherimport\"\n"},
		{"vsfix.\u26a1\ufe0f", "package p\n\nimport _ \"logosvsimport\"\n"},
	}
	for _, f := range files {
		if err := os.WriteFile(filepath.Join(pkgdir, f.name), []byte(f.src), 0o666); err != nil {
			t.Fatal(err)
		}
	}
	pkg := indexPackageForDir(t, modroot, "p")

	wantImports := []string{
		"goimport",
		"logosaliasimport",
		"logosgopherimport",
		"logosimport",
		"logosscrollimport",
		"logosvsimport",
	}
	wantTestImports := []string{"logostestimport"}
	wantPlainImports := []string{"goimport"}

	t.Run("logos", func(t *testing.T) {
		setLogosMode(t)
		got, gotTest, err := pkg.ScanDir(imports.AnyTags())
		if err != nil {
			t.Fatalf("ScanDir: %v", err)
		}
		if !reflect.DeepEqual(got, wantImports) {
			t.Errorf("ScanDir imports = %q, want %q", got, wantImports)
		}
		if !reflect.DeepEqual(gotTest, wantTestImports) {
			t.Errorf("ScanDir test imports = %q, want %q", gotTest, wantTestImports)
		}
	})

	plainTagSets := map[string]map[string]bool{
		"plainAnyTags":          {"*": true},
		"plainExplicitLogosTag": {"*": true, "logos": true},
	}
	for name, tags := range plainTagSets {
		t.Run(name, func(t *testing.T) {
			setPlainGoMode(t)
			got, gotTest, err := pkg.ScanDir(tags)
			if err != nil {
				t.Fatalf("ScanDir: %v", err)
			}
			if !reflect.DeepEqual(got, wantPlainImports) {
				t.Errorf("ScanDir imports = %q, want %q", got, wantPlainImports)
			}
			if len(gotTest) != 0 {
				t.Errorf("ScanDir test imports = %q, want none", gotTest)
			}
		})
	}
}

func TestScanDirLogosOnlyIndexed(t *testing.T) {
	modroot := t.TempDir()
	pkgdir := filepath.Join(modroot, "only")
	if err := os.MkdirAll(pkgdir, 0o777); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(pkgdir, "only.lgo"), []byte("package only\n\nimport _ \"logosonlyimport\"\n"), 0o666); err != nil {
		t.Fatal(err)
	}
	pkg := indexPackageForDir(t, modroot, "only")

	setPlainGoMode(t)
	_, _, err := pkg.ScanDir(imports.AnyTags())
	if !errors.Is(err, imports.ErrNoGo) {
		t.Fatalf("plain ScanDir err = %v, want ErrNoGo", err)
	}

	setLogosMode(t)
	got, _, err := pkg.ScanDir(imports.AnyTags())
	if err != nil {
		t.Fatalf("ScanDir: %v", err)
	}
	want := []string{"logosonlyimport"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("ScanDir imports = %q, want %q", got, want)
	}
}

func TestScanDirLogosGetPackage(t *testing.T) {
	modroot := t.TempDir()
	pkgdir := filepath.Join(modroot, "gp")
	if err := os.MkdirAll(pkgdir, 0o777); err != nil {
		t.Fatal(err)
	}
	files := []struct{ name, src string }{
		{"g.go", "package gp\n\nimport _ \"goimport\"\n"},
		{"g.lgo", "package gp\n\nimport _ \"logosgpimport\"\n"},
	}
	aged := time.Now().Add(-3 * modTimeCutoff)
	for _, f := range files {
		path := filepath.Join(pkgdir, f.name)
		if err := os.WriteFile(path, []byte(f.src), 0o666); err != nil {
			t.Fatal(err)
		}
		if err := os.Chtimes(path, aged, aged); err != nil {
			t.Fatal(err)
		}
	}

	setPlainGoMode(t)
	pkg, err := GetPackage(modroot, pkgdir)
	if err != nil {
		t.Fatalf("GetPackage: %v", err)
	}
	got, gotTest, err := pkg.ScanDir(imports.AnyTags())
	if err != nil {
		t.Fatalf("ScanDir: %v", err)
	}
	wantPlain := []string{"goimport"}
	if !reflect.DeepEqual(got, wantPlain) {
		t.Errorf("plain ScanDir imports = %q, want %q", got, wantPlain)
	}
	if len(gotTest) != 0 {
		t.Errorf("plain ScanDir test imports = %q, want none", gotTest)
	}

	setLogosMode(t)
	got, _, err = pkg.ScanDir(imports.AnyTags())
	if err != nil {
		t.Fatalf("ScanDir: %v", err)
	}
	wantLogos := []string{"goimport", "logosgpimport"}
	if !reflect.DeepEqual(got, wantLogos) {
		t.Errorf("logos ScanDir imports = %q, want %q", got, wantLogos)
	}
}
