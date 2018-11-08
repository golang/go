// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modcmd

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/modload"
	"cmd/go/internal/module"
)

var cmdVendor = &base.Command{
	UsageLine: "go mod vendor [-v]",
	Short:     "make vendored copy of dependencies",
	Long: `
Vendor resets the main module's vendor directory to include all packages
needed to build and test all the main module's packages.
It does not include test code for vendored packages.

The -v flag causes vendor to print the names of vendored
modules and packages to standard error.
	`,
	Run: runVendor,
}

func init() {
	cmdVendor.Flag.BoolVar(&cfg.BuildV, "v", false, "")
}

func runVendor(cmd *base.Command, args []string) {
	if len(args) != 0 {
		base.Fatalf("go mod vendor: vendor takes no arguments")
	}
	pkgs := modload.LoadVendor()

	vdir := filepath.Join(modload.ModRoot(), "vendor")
	if err := os.RemoveAll(vdir); err != nil {
		base.Fatalf("go mod vendor: %v", err)
	}

	modpkgs := make(map[module.Version][]string)
	for _, pkg := range pkgs {
		m := modload.PackageModule(pkg)
		if m == modload.Target {
			continue
		}
		modpkgs[m] = append(modpkgs[m], pkg)
	}

	var buf bytes.Buffer
	for _, m := range modload.BuildList()[1:] {
		if pkgs := modpkgs[m]; len(pkgs) > 0 {
			repl := ""
			if r := modload.Replacement(m); r.Path != "" {
				repl = " => " + r.Path
				if r.Version != "" {
					repl += " " + r.Version
				}
			}
			fmt.Fprintf(&buf, "# %s %s%s\n", m.Path, m.Version, repl)
			if cfg.BuildV {
				fmt.Fprintf(os.Stderr, "# %s %s%s\n", m.Path, m.Version, repl)
			}
			for _, pkg := range pkgs {
				fmt.Fprintf(&buf, "%s\n", pkg)
				if cfg.BuildV {
					fmt.Fprintf(os.Stderr, "%s\n", pkg)
				}
				vendorPkg(vdir, pkg)
			}
		}
	}
	if buf.Len() == 0 {
		fmt.Fprintf(os.Stderr, "go: no dependencies to vendor\n")
		return
	}
	if err := ioutil.WriteFile(filepath.Join(vdir, "modules.txt"), buf.Bytes(), 0666); err != nil {
		base.Fatalf("go mod vendor: %v", err)
	}
}

func vendorPkg(vdir, pkg string) {
	realPath := modload.ImportMap(pkg)
	if realPath != pkg && modload.ImportMap(realPath) != "" {
		fmt.Fprintf(os.Stderr, "warning: %s imported as both %s and %s; making two copies.\n", realPath, realPath, pkg)
	}

	dst := filepath.Join(vdir, pkg)
	src := modload.PackageDir(realPath)
	if src == "" {
		fmt.Fprintf(os.Stderr, "internal error: no pkg for %s -> %s\n", pkg, realPath)
	}
	copyDir(dst, src, matchNonTest)
	if m := modload.PackageModule(realPath); m.Path != "" {
		copyMetadata(m.Path, realPath, dst, src)
	}
}

type metakey struct {
	modPath string
	dst     string
}

var copiedMetadata = make(map[metakey]bool)

// copyMetadata copies metadata files from parents of src to parents of dst,
// stopping after processing the src parent for modPath.
func copyMetadata(modPath, pkg, dst, src string) {
	for parent := 0; ; parent++ {
		if copiedMetadata[metakey{modPath, dst}] {
			break
		}
		copiedMetadata[metakey{modPath, dst}] = true
		if parent > 0 {
			copyDir(dst, src, matchMetadata)
		}
		if modPath == pkg {
			break
		}
		pkg = filepath.Dir(pkg)
		dst = filepath.Dir(dst)
		src = filepath.Dir(src)
	}
}

// metaPrefixes is the list of metadata file prefixes.
// Vendoring copies metadata files from parents of copied directories.
// Note that this list could be arbitrarily extended, and it is longer
// in other tools (such as godep or dep). By using this limited set of
// prefixes and also insisting on capitalized file names, we are trying
// to nudge people toward more agreement on the naming
// and also trying to avoid false positives.
var metaPrefixes = []string{
	"AUTHORS",
	"CONTRIBUTORS",
	"COPYLEFT",
	"COPYING",
	"COPYRIGHT",
	"LEGAL",
	"LICENSE",
	"NOTICE",
	"PATENTS",
}

// matchMetadata reports whether info is a metadata file.
func matchMetadata(info os.FileInfo) bool {
	name := info.Name()
	for _, p := range metaPrefixes {
		if strings.HasPrefix(name, p) {
			return true
		}
	}
	return false
}

// matchNonTest reports whether info is any non-test file (including non-Go files).
func matchNonTest(info os.FileInfo) bool {
	return !strings.HasSuffix(info.Name(), "_test.go")
}

// copyDir copies all regular files satisfying match(info) from src to dst.
func copyDir(dst, src string, match func(os.FileInfo) bool) {
	files, err := ioutil.ReadDir(src)
	if err != nil {
		base.Fatalf("go mod vendor: %v", err)
	}
	if err := os.MkdirAll(dst, 0777); err != nil {
		base.Fatalf("go mod vendor: %v", err)
	}
	for _, file := range files {
		if file.IsDir() || !file.Mode().IsRegular() || !match(file) {
			continue
		}
		r, err := os.Open(filepath.Join(src, file.Name()))
		if err != nil {
			base.Fatalf("go mod vendor: %v", err)
		}
		w, err := os.Create(filepath.Join(dst, file.Name()))
		if err != nil {
			base.Fatalf("go mod vendor: %v", err)
		}
		if _, err := io.Copy(w, r); err != nil {
			base.Fatalf("go mod vendor: %v", err)
		}
		r.Close()
		if err := w.Close(); err != nil {
			base.Fatalf("go mod vendor: %v", err)
		}
	}
}
