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
	"sort"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/imports"
	"cmd/go/internal/modload"
	"cmd/go/internal/work"

	"golang.org/x/mod/module"
	"golang.org/x/mod/semver"
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
	work.AddModCommonFlags(cmdVendor)
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

	includeAllReplacements := false
	isExplicit := map[module.Version]bool{}
	if gv := modload.ModFile().Go; gv != nil && semver.Compare("v"+gv.Version, "v1.14") >= 0 {
		// If the Go version is at least 1.14, annotate all explicit 'require' and
		// 'replace' targets found in the go.mod file so that we can perform a
		// stronger consistency check when -mod=vendor is set.
		for _, r := range modload.ModFile().Require {
			isExplicit[r.Mod] = true
		}
		includeAllReplacements = true
	}

	var buf bytes.Buffer
	for _, m := range modload.BuildList()[1:] {
		if pkgs := modpkgs[m]; len(pkgs) > 0 || isExplicit[m] {
			line := moduleLine(m, modload.Replacement(m))
			buf.WriteString(line)
			if cfg.BuildV {
				os.Stderr.WriteString(line)
			}
			if isExplicit[m] {
				buf.WriteString("## explicit\n")
				if cfg.BuildV {
					os.Stderr.WriteString("## explicit\n")
				}
			}
			sort.Strings(pkgs)
			for _, pkg := range pkgs {
				fmt.Fprintf(&buf, "%s\n", pkg)
				if cfg.BuildV {
					fmt.Fprintf(os.Stderr, "%s\n", pkg)
				}
				vendorPkg(vdir, pkg)
			}
		}
	}

	if includeAllReplacements {
		// Record unused and wildcard replacements at the end of the modules.txt file:
		// without access to the complete build list, the consumer of the vendor
		// directory can't otherwise determine that those replacements had no effect.
		for _, r := range modload.ModFile().Replace {
			if len(modpkgs[r.Old]) > 0 {
				// We we already recorded this replacement in the entry for the replaced
				// module with the packages it provides.
				continue
			}

			line := moduleLine(r.Old, r.New)
			buf.WriteString(line)
			if cfg.BuildV {
				os.Stderr.WriteString(line)
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

func moduleLine(m, r module.Version) string {
	b := new(strings.Builder)
	b.WriteString("# ")
	b.WriteString(m.Path)
	if m.Version != "" {
		b.WriteString(" ")
		b.WriteString(m.Version)
	}
	if r.Path != "" {
		b.WriteString(" => ")
		b.WriteString(r.Path)
		if r.Version != "" {
			b.WriteString(" ")
			b.WriteString(r.Version)
		}
	}
	b.WriteString("\n")
	return b.String()
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
	copyDir(dst, src, matchPotentialSourceFile)
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
func matchMetadata(dir string, info os.FileInfo) bool {
	name := info.Name()
	for _, p := range metaPrefixes {
		if strings.HasPrefix(name, p) {
			return true
		}
	}
	return false
}

// matchPotentialSourceFile reports whether info may be relevant to a build operation.
func matchPotentialSourceFile(dir string, info os.FileInfo) bool {
	if strings.HasSuffix(info.Name(), "_test.go") {
		return false
	}
	if strings.HasSuffix(info.Name(), ".go") {
		f, err := os.Open(filepath.Join(dir, info.Name()))
		if err != nil {
			base.Fatalf("go mod vendor: %v", err)
		}
		defer f.Close()

		content, err := imports.ReadImports(f, false, nil)
		if err == nil && !imports.ShouldBuild(content, imports.AnyTags()) {
			// The file is explicitly tagged "ignore", so it can't affect the build.
			// Leave it out.
			return false
		}
		return true
	}

	// We don't know anything about this file, so optimistically assume that it is
	// needed.
	return true
}

// copyDir copies all regular files satisfying match(info) from src to dst.
func copyDir(dst, src string, match func(dir string, info os.FileInfo) bool) {
	files, err := ioutil.ReadDir(src)
	if err != nil {
		base.Fatalf("go mod vendor: %v", err)
	}
	if err := os.MkdirAll(dst, 0777); err != nil {
		base.Fatalf("go mod vendor: %v", err)
	}
	for _, file := range files {
		if file.IsDir() || !file.Mode().IsRegular() || !match(src, file) {
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
