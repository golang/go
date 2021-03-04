// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modcmd

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"go/build"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/fsys"
	"cmd/go/internal/imports"
	"cmd/go/internal/load"
	"cmd/go/internal/modload"
	"cmd/go/internal/str"

	"golang.org/x/mod/module"
	"golang.org/x/mod/semver"
)

var cmdVendor = &base.Command{
	UsageLine: "go mod vendor [-e] [-v]",
	Short:     "make vendored copy of dependencies",
	Long: `
Vendor resets the main module's vendor directory to include all packages
needed to build and test all the main module's packages.
It does not include test code for vendored packages.

The -v flag causes vendor to print the names of vendored
modules and packages to standard error.

The -e flag causes vendor to attempt to proceed despite errors
encountered while loading packages.

See https://golang.org/ref/mod#go-mod-vendor for more about 'go mod vendor'.
	`,
	Run: runVendor,
}

var vendorE bool // if true, report errors but proceed anyway

func init() {
	cmdVendor.Flag.BoolVar(&cfg.BuildV, "v", false, "")
	cmdVendor.Flag.BoolVar(&vendorE, "e", false, "")
	base.AddModCommonFlags(&cmdVendor.Flag)
}

func runVendor(ctx context.Context, cmd *base.Command, args []string) {
	if len(args) != 0 {
		base.Fatalf("go mod vendor: vendor takes no arguments")
	}
	modload.ForceUseModules = true
	modload.RootMode = modload.NeedRoot

	loadOpts := modload.PackageOpts{
		Tags:                     imports.AnyTags(),
		ResolveMissingImports:    true,
		UseVendorAll:             true,
		AllowErrors:              vendorE,
		SilenceMissingStdImports: true,
	}
	_, pkgs := modload.LoadPackages(ctx, loadOpts, "all")

	vdir := filepath.Join(modload.ModRoot(), "vendor")
	if err := os.RemoveAll(vdir); err != nil {
		base.Fatalf("go mod vendor: %v", err)
	}

	modpkgs := make(map[module.Version][]string)
	for _, pkg := range pkgs {
		m := modload.PackageModule(pkg)
		if m.Path == "" || m == modload.Target {
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

	var vendorMods []module.Version
	for m := range isExplicit {
		vendorMods = append(vendorMods, m)
	}
	for m := range modpkgs {
		if !isExplicit[m] {
			vendorMods = append(vendorMods, m)
		}
	}
	module.Sort(vendorMods)

	var buf bytes.Buffer
	for _, m := range vendorMods {
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
		pkgs := modpkgs[m]
		sort.Strings(pkgs)
		for _, pkg := range pkgs {
			fmt.Fprintf(&buf, "%s\n", pkg)
			if cfg.BuildV {
				fmt.Fprintf(os.Stderr, "%s\n", pkg)
			}
			vendorPkg(vdir, pkg)
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

	if err := os.MkdirAll(vdir, 0777); err != nil {
		base.Fatalf("go mod vendor: %v", err)
	}

	if err := os.WriteFile(filepath.Join(vdir, "modules.txt"), buf.Bytes(), 0666); err != nil {
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
	// TODO(#42504): Instead of calling modload.ImportMap then build.ImportDir,
	// just call load.PackagesAndErrors. To do that, we need to add a good way
	// to ignore build constraints.
	realPath := modload.ImportMap(pkg)
	if realPath != pkg && modload.ImportMap(realPath) != "" {
		fmt.Fprintf(os.Stderr, "warning: %s imported as both %s and %s; making two copies.\n", realPath, realPath, pkg)
	}

	copiedFiles := make(map[string]bool)
	dst := filepath.Join(vdir, pkg)
	src := modload.PackageDir(realPath)
	if src == "" {
		fmt.Fprintf(os.Stderr, "internal error: no pkg for %s -> %s\n", pkg, realPath)
	}
	copyDir(dst, src, matchPotentialSourceFile, copiedFiles)
	if m := modload.PackageModule(realPath); m.Path != "" {
		copyMetadata(m.Path, realPath, dst, src, copiedFiles)
	}

	ctx := build.Default
	ctx.UseAllFiles = true
	bp, err := ctx.ImportDir(src, build.IgnoreVendor)
	// Because UseAllFiles is set on the build.Context, it's possible ta get
	// a MultiplePackageError on an otherwise valid package: the package could
	// have different names for GOOS=windows and GOOS=mac for example. On the
	// other hand if there's a NoGoError, the package might have source files
	// specifying "// +build ignore" those packages should be skipped because
	// embeds from ignored files can't be used.
	// TODO(#42504): Find a better way to avoid errors from ImportDir. We'll
	// need to figure this out when we switch to PackagesAndErrors as per the
	// TODO above.
	var multiplePackageError *build.MultiplePackageError
	var noGoError *build.NoGoError
	if err != nil {
		if errors.As(err, &noGoError) {
			return // No source files in this package are built. Skip embeds in ignored files.
		} else if !errors.As(err, &multiplePackageError) { // multiplePackgeErrors are okay, but others are not.
			base.Fatalf("internal error: failed to find embedded files of %s: %v\n", pkg, err)
		}
	}
	embedPatterns := str.StringList(bp.EmbedPatterns, bp.TestEmbedPatterns, bp.XTestEmbedPatterns)
	embeds, err := load.ResolveEmbed(bp.Dir, embedPatterns)
	if err != nil {
		base.Fatalf("go mod vendor: %v", err)
	}
	for _, embed := range embeds {
		embedDst := filepath.Join(dst, embed)
		if copiedFiles[embedDst] {
			continue
		}

		// Copy the file as is done by copyDir below.
		r, err := os.Open(filepath.Join(src, embed))
		if err != nil {
			base.Fatalf("go mod vendor: %v", err)
		}
		if err := os.MkdirAll(filepath.Dir(embedDst), 0777); err != nil {
			base.Fatalf("go mod vendor: %v", err)
		}
		w, err := os.Create(embedDst)
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

type metakey struct {
	modPath string
	dst     string
}

var copiedMetadata = make(map[metakey]bool)

// copyMetadata copies metadata files from parents of src to parents of dst,
// stopping after processing the src parent for modPath.
func copyMetadata(modPath, pkg, dst, src string, copiedFiles map[string]bool) {
	for parent := 0; ; parent++ {
		if copiedMetadata[metakey{modPath, dst}] {
			break
		}
		copiedMetadata[metakey{modPath, dst}] = true
		if parent > 0 {
			copyDir(dst, src, matchMetadata, copiedFiles)
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
func matchMetadata(dir string, info fs.DirEntry) bool {
	name := info.Name()
	for _, p := range metaPrefixes {
		if strings.HasPrefix(name, p) {
			return true
		}
	}
	return false
}

// matchPotentialSourceFile reports whether info may be relevant to a build operation.
func matchPotentialSourceFile(dir string, info fs.DirEntry) bool {
	if strings.HasSuffix(info.Name(), "_test.go") {
		return false
	}
	if strings.HasSuffix(info.Name(), ".go") {
		f, err := fsys.Open(filepath.Join(dir, info.Name()))
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
func copyDir(dst, src string, match func(dir string, info fs.DirEntry) bool, copiedFiles map[string]bool) {
	files, err := os.ReadDir(src)
	if err != nil {
		base.Fatalf("go mod vendor: %v", err)
	}
	if err := os.MkdirAll(dst, 0777); err != nil {
		base.Fatalf("go mod vendor: %v", err)
	}
	for _, file := range files {
		if file.IsDir() || !file.Type().IsRegular() || !match(src, file) {
			continue
		}
		copiedFiles[file.Name()] = true
		r, err := os.Open(filepath.Join(src, file.Name()))
		if err != nil {
			base.Fatalf("go mod vendor: %v", err)
		}
		dstPath := filepath.Join(dst, file.Name())
		copiedFiles[dstPath] = true
		w, err := os.Create(dstPath)
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
