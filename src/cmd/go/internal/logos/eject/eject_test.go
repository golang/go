// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eject

import (
	"os"
	"path/filepath"
	"testing"

	"cmd/go/internal/load"
	"cmd/go/internal/modinfo"
)

func TestEmitPackagePlatformFiles(t *testing.T) {
	modDir := t.TempDir()
	outDir := t.TempDir()
	staging := t.TempDir()

	pkgDir := filepath.Join(modDir, "plat")
	if err := os.MkdirAll(pkgDir, 0777); err != nil {
		t.Fatal(err)
	}

	if err := os.WriteFile(filepath.Join(pkgDir, "main.lgo"), []byte("package main\n\nfunc main() {}\n"), 0666); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(pkgDir, "plat_linux.lgo"), []byte("package main\n\nfunc Platform() string { return \"linux\" }\n"), 0666); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(pkgDir, "plat_windows.lgo"), []byte("package main\n\nfunc Platform() string { return \"windows\" }\n"), 0666); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(pkgDir, "plat_notlinux.lgo"), []byte("//go:build !linux\n\npackage main\n\nfunc NotLinux() {}\n"), 0666); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(pkgDir, "plat_ignore.lgo"), []byte("//go:build ignore\n\npackage main\n\nfunc Ignored() {}\n"), 0666); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(pkgDir, "plus_ignore.lgo"), []byte("// +build ignore\n\npackage main\n\nfunc PlusIgnored() {}\n"), 0666); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(pkgDir, "cgo_helper.lgo"), []byte("package main\n\nfunc CgoHelper() {}\n"), 0666); err != nil {
		t.Fatal(err)
	}

	stagingPkg := filepath.Join(staging, "example.com/plat")
	if err := os.MkdirAll(stagingPkg, 0777); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(stagingPkg, "main.go"), []byte("package main\n\nfunc main() {}\n"), 0666); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(stagingPkg, "plat_linux.go"), []byte("package main\n\nfunc Platform() string { return \"linux\" }\n"), 0666); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(stagingPkg, "plat_windows.go"), []byte("package main\n\nfunc Platform() string { return \"windows\" }\n"), 0666); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(stagingPkg, "plat_notlinux.go"), []byte("//go:build !linux\n\npackage main\n\nfunc NotLinux() {}\n"), 0666); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(stagingPkg, "cgo_helper.go"), []byte("package main\n\nfunc CgoHelper() {}\n"), 0666); err != nil {
		t.Fatal(err)
	}

	p := &load.Package{
		PackagePublic: load.PackagePublic{
			ImportPath:     "example.com/plat",
			Name:           "main",
			Dir:            pkgDir,
			Module:         &modinfo.ModulePublic{Path: "example.com/plat", Main: true, Dir: modDir},
			GoFiles:        []string{"main.lgo", "plat_linux.lgo"},
			CgoFiles:       []string{"cgo_helper.lgo"},
			IgnoredGoFiles: []string{"plat_windows.lgo", "plat_notlinux.lgo", "plat_ignore.lgo", "plus_ignore.lgo"},
		},
	}

	plan := &ejectPlan{
		main:      &ejectModule{dir: modDir, pkgs: []*load.Package{p}},
		testBuilt: map[*load.Package]bool{},
	}

	emitPackageGoFiles(p, plan, staging, outDir, modDir)

	if _, err := os.Stat(filepath.Join(outDir, "plat", "main.go")); err != nil {
		t.Errorf("main.go missing: %v", err)
	}
	if _, err := os.Stat(filepath.Join(outDir, "plat", "plat_linux.go")); err != nil {
		t.Errorf("plat_linux.go missing: %v", err)
	}
	if _, err := os.Stat(filepath.Join(outDir, "plat", "plat_windows.go")); err != nil {
		t.Errorf("plat_windows.go missing: %v", err)
	}
	if _, err := os.Stat(filepath.Join(outDir, "plat", "plat_notlinux.go")); err != nil {
		t.Errorf("plat_notlinux.go missing: %v", err)
	}
	if _, err := os.Stat(filepath.Join(outDir, "plat", "cgo_helper.go")); err != nil {
		t.Errorf("cgo_helper.go missing: %v", err)
	}
	if _, err := os.Stat(filepath.Join(outDir, "plat", "plat_ignore.go")); !os.IsNotExist(err) {
		t.Errorf("plat_ignore.go should not be emitted, stat err: %v", err)
	}
	if _, err := os.Stat(filepath.Join(outDir, "plat", "plus_ignore.go")); !os.IsNotExist(err) {
		t.Errorf("plus_ignore.go should not be emitted, stat err: %v", err)
	}
}

func TestFindVariantContextPlatformLogosSyntax(t *testing.T) {
	modDir := t.TempDir()
	pkgDir := filepath.Join(modDir, "plat")
	if err := os.MkdirAll(pkgDir, 0777); err != nil {
		t.Fatal(err)
	}
	linuxSrc := "package main\n\nfunc Platform() string {\n\tvar u *User? = &User{Name: \"linux\"}\n\treturn (u?.Name ?? \"fallback\")\n}\n"
	windowsSrc := "package main\n\nfunc Platform() string {\n\tvar u *User? = &User{Name: \"windows\"}\n\treturn (u?.Name ?? \"fallback\")\n}\n"
	arm64Src := "package main\n\nfunc PlatformARM64() string {\n\tvar u *User? = &User{Name: \"windows-arm64\"}\n\treturn (u?.Name ?? \"fallback\")\n}\n"
	for name, src := range map[string]string{
		"plat_linux.lgo":         linuxSrc,
		"plat_windows.lgo":       windowsSrc,
		"plat_windows_arm64.lgo": arm64Src,
	} {
		if err := os.WriteFile(filepath.Join(pkgDir, name), []byte(src), 0666); err != nil {
			t.Fatal(err)
		}
	}
	if goos, goarch, _, ok := findVariantContextForFile(pkgDir, "plat_windows.lgo", "linux", "amd64", nil); !ok || goos != "windows" {
		t.Errorf("plat_windows.lgo context = %q/%q ok=%v, want windows", goos, goarch, ok)
	}
	if goos, goarch, _, ok := findVariantContextForFile(pkgDir, "plat_windows_arm64.lgo", "linux", "amd64", nil); !ok || goos != "windows" || goarch != "arm64" {
		t.Errorf("plat_windows_arm64.lgo context = %q/%q ok=%v, want windows/arm64", goos, goarch, ok)
	}
}
