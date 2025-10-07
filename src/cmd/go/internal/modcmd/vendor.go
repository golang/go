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
	"path"
	"path/filepath"
	"slices"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/fsys"
	"cmd/go/internal/gover"
	"cmd/go/internal/imports"
	"cmd/go/internal/load"
	"cmd/go/internal/modload"
	"cmd/go/internal/str"

	"golang.org/x/mod/module"
)

var cmdVendor = &base.Command{
	UsageLine: "go mod vendor [-e] [-v] [-o outdir]",
	Short:     "make vendored copy of dependencies",
	Long: `
Vendor resets the main module's vendor directory to include all packages
needed to build and test all the main module's packages.
It does not include test code for vendored packages.

The -v flag causes vendor to print the names of vendored
modules and packages to standard error.

The -e flag causes vendor to attempt to proceed despite errors
encountered while loading packages.

The -o flag causes vendor to create the vendor directory at the given
path instead of "vendor". The go command can only use a vendor directory
named "vendor" within the module root directory, so this flag is
primarily useful for other tools.

See https://golang.org/ref/mod#go-mod-vendor for more about 'go mod vendor'.
	`,
	Run: runVendor,
}

var vendorE bool   // if true, report errors but proceed anyway
var vendorO string // if set, overrides the default output directory

func init() {
	cmdVendor.Flag.BoolVar(&cfg.BuildV, "v", false, "")
	cmdVendor.Flag.BoolVar(&vendorE, "e", false, "")
	cmdVendor.Flag.StringVar(&vendorO, "o", "", "")
	base.AddChdirFlag(&cmdVendor.Flag)
	base.AddModCommonFlags(&cmdVendor.Flag)
}

func runVendor(ctx context.Context, cmd *base.Command, args []string) {
	modload.InitWorkfile()
	if modload.WorkFilePath() != "" {
		base.Fatalf("go: 'go mod vendor' cannot be run in workspace mode. Run 'go work vendor' to vendor the workspace or set 'GOWORK=off' to exit workspace mode.")
	}
	RunVendor(ctx, vendorE, vendorO, args)
}

func RunVendor(ctx context.Context, vendorE bool, vendorO string, args []string) {
	if len(args) != 0 {
		base.Fatalf("go: 'go mod vendor' accepts no arguments")
	}
	modload.LoaderState.ForceUseModules = true
	modload.LoaderState.RootMode = modload.NeedRoot

	loadOpts := modload.PackageOpts{
		Tags:                     imports.AnyTags(),
		VendorModulesInGOROOTSrc: true,
		ResolveMissingImports:    true,
		UseVendorAll:             true,
		AllowErrors:              vendorE,
		SilenceMissingStdImports: true,
	}
	_, pkgs := modload.LoadPackages(ctx, loadOpts, "all")

	var vdir string
	switch {
	case filepath.IsAbs(vendorO):
		vdir = vendorO
	case vendorO != "":
		vdir = filepath.Join(base.Cwd(), vendorO)
	default:
		vdir = filepath.Join(modload.VendorDir())
	}
	if err := os.RemoveAll(vdir); err != nil {
		base.Fatal(err)
	}

	modpkgs := make(map[module.Version][]string)
	for _, pkg := range pkgs {
		m := modload.PackageModule(pkg)
		if m.Path == "" || modload.MainModules.Contains(m.Path) {
			continue
		}
		modpkgs[m] = append(modpkgs[m], pkg)
	}
	checkPathCollisions(modpkgs)

	includeAllReplacements := false
	includeGoVersions := false
	isExplicit := map[module.Version]bool{}
	gv := modload.MainModules.GoVersion()
	if gover.Compare(gv, "1.14") >= 0 && (modload.FindGoWork(base.Cwd()) != "" || modload.ModFile().Go != nil) {
		// If the Go version is at least 1.14, annotate all explicit 'require' and
		// 'replace' targets found in the go.mod file so that we can perform a
		// stronger consistency check when -mod=vendor is set.
		for _, m := range modload.MainModules.Versions() {
			if modFile := modload.MainModules.ModFile(m); modFile != nil {
				for _, r := range modFile.Require {
					isExplicit[r.Mod] = true
				}
			}

		}
		includeAllReplacements = true
	}
	if gover.Compare(gv, "1.17") >= 0 {
		// If the Go version is at least 1.17, annotate all modules with their
		// 'go' version directives.
		includeGoVersions = true
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
	gover.ModSort(vendorMods)

	var (
		buf bytes.Buffer
		w   io.Writer = &buf
	)
	if cfg.BuildV {
		w = io.MultiWriter(&buf, os.Stderr)
	}

	if modload.MainModules.WorkFile() != nil {
		fmt.Fprintf(w, "## workspace\n")
	}

	replacementWritten := make(map[module.Version]bool)
	for _, m := range vendorMods {
		replacement := modload.Replacement(m)
		line := moduleLine(m, replacement)
		replacementWritten[m] = true
		io.WriteString(w, line)

		goVersion := ""
		if includeGoVersions {
			goVersion = modload.ModuleInfo(ctx, m.Path).GoVersion
		}
		switch {
		case isExplicit[m] && goVersion != "":
			fmt.Fprintf(w, "## explicit; go %s\n", goVersion)
		case isExplicit[m]:
			io.WriteString(w, "## explicit\n")
		case goVersion != "":
			fmt.Fprintf(w, "## go %s\n", goVersion)
		}

		pkgs := modpkgs[m]
		slices.Sort(pkgs)
		for _, pkg := range pkgs {
			fmt.Fprintf(w, "%s\n", pkg)
			vendorPkg(vdir, pkg)
		}
	}

	if includeAllReplacements {
		// Record unused and wildcard replacements at the end of the modules.txt file:
		// without access to the complete build list, the consumer of the vendor
		// directory can't otherwise determine that those replacements had no effect.
		for _, m := range modload.MainModules.Versions() {
			if workFile := modload.MainModules.WorkFile(); workFile != nil {
				for _, r := range workFile.Replace {
					if replacementWritten[r.Old] {
						// We already recorded this replacement.
						continue
					}
					replacementWritten[r.Old] = true

					line := moduleLine(r.Old, r.New)
					buf.WriteString(line)
					if cfg.BuildV {
						os.Stderr.WriteString(line)
					}
				}
			}
			if modFile := modload.MainModules.ModFile(m); modFile != nil {
				for _, r := range modFile.Replace {
					if replacementWritten[r.Old] {
						// We already recorded this replacement.
						continue
					}
					replacementWritten[r.Old] = true
					rNew := modload.Replacement(r.Old)
					if rNew == (module.Version{}) {
						// There is no replacement. Don't try to write it.
						continue
					}

					line := moduleLine(r.Old, rNew)
					buf.WriteString(line)
					if cfg.BuildV {
						os.Stderr.WriteString(line)
					}
				}
			}
		}
	}

	if buf.Len() == 0 {
		fmt.Fprintf(os.Stderr, "go: no dependencies to vendor\n")
		return
	}

	if err := os.MkdirAll(vdir, 0777); err != nil {
		base.Fatal(err)
	}

	if err := os.WriteFile(filepath.Join(vdir, "modules.txt"), buf.Bytes(), 0666); err != nil {
		base.Fatal(err)
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
		if str.HasFilePathPrefix(filepath.Clean(r.Path), "vendor") {
			base.Fatalf("go: replacement path %s inside vendor directory", r.Path)
		}
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
	src, realPath, _ := modload.Lookup("", false, pkg)
	if src == "" {
		base.Errorf("internal error: no pkg for %s\n", pkg)
		return
	}
	if realPath != pkg {
		// TODO(#26904): Revisit whether this behavior still makes sense.
		// This should actually be impossible today, because the import map is the
		// identity function for packages outside of the standard library.
		//
		// Part of the purpose of the vendor directory is to allow the packages in
		// the module to continue to build in GOPATH mode, and GOPATH-mode users
		// won't know about replacement aliasing. How important is it to maintain
		// compatibility?
		fmt.Fprintf(os.Stderr, "warning: %s imported as both %s and %s; making two copies.\n", realPath, realPath, pkg)
	}

	copiedFiles := make(map[string]bool)
	dst := filepath.Join(vdir, pkg)
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
	// specifying "//go:build ignore" those packages should be skipped because
	// embeds from ignored files can't be used.
	// TODO(#42504): Find a better way to avoid errors from ImportDir. We'll
	// need to figure this out when we switch to PackagesAndErrors as per the
	// TODO above.
	var multiplePackageError *build.MultiplePackageError
	var noGoError *build.NoGoError
	if err != nil {
		if errors.As(err, &noGoError) {
			return // No source files in this package are built. Skip embeds in ignored files.
		} else if !errors.As(err, &multiplePackageError) { // multiplePackageErrors are OK, but others are not.
			base.Fatalf("internal error: failed to find embedded files of %s: %v\n", pkg, err)
		}
	}
	var embedPatterns []string
	if gover.Compare(modload.MainModules.GoVersion(), "1.22") >= 0 {
		embedPatterns = bp.EmbedPatterns
	} else {
		// Maintain the behavior of https://github.com/golang/go/issues/63473
		// so that we continue to agree with older versions of the go command
		// about the contents of vendor directories in existing modules
		embedPatterns = str.StringList(bp.EmbedPatterns, bp.TestEmbedPatterns, bp.XTestEmbedPatterns)
	}
	embeds, err := load.ResolveEmbed(bp.Dir, embedPatterns)
	if err != nil {
		format := "go: resolving embeds in %s: %v\n"
		if vendorE {
			fmt.Fprintf(os.Stderr, format, pkg, err)
		} else {
			base.Errorf(format, pkg, err)
		}
		return
	}
	for _, embed := range embeds {
		embedDst := filepath.Join(dst, embed)
		if copiedFiles[embedDst] {
			continue
		}

		// Copy the file as is done by copyDir below.
		err := func() error {
			r, err := os.Open(filepath.Join(src, embed))
			if err != nil {
				return err
			}
			if err := os.MkdirAll(filepath.Dir(embedDst), 0777); err != nil {
				return err
			}
			w, err := os.Create(embedDst)
			if err != nil {
				return err
			}
			if _, err := io.Copy(w, r); err != nil {
				return err
			}
			r.Close()
			return w.Close()
		}()
		if err != nil {
			if vendorE {
				fmt.Fprintf(os.Stderr, "go: %v\n", err)
			} else {
				base.Error(err)
			}
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
		pkg = path.Dir(pkg)
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
	if info.Name() == "go.mod" || info.Name() == "go.sum" {
		if gv := modload.MainModules.GoVersion(); gover.Compare(gv, "1.17") >= 0 {
			// As of Go 1.17, we strip go.mod and go.sum files from dependency modules.
			// Otherwise, 'go' commands invoked within the vendor subtree may misidentify
			// an arbitrary directory within the vendor tree as a module root.
			// (See https://golang.org/issue/42970.)
			return false
		}
	}
	if strings.HasSuffix(info.Name(), ".go") {
		f, err := fsys.Open(filepath.Join(dir, info.Name()))
		if err != nil {
			base.Fatal(err)
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
		base.Fatal(err)
	}
	if err := os.MkdirAll(dst, 0777); err != nil {
		base.Fatal(err)
	}
	for _, file := range files {
		if file.IsDir() || !file.Type().IsRegular() || !match(src, file) {
			continue
		}
		copiedFiles[file.Name()] = true
		r, err := os.Open(filepath.Join(src, file.Name()))
		if err != nil {
			base.Fatal(err)
		}
		dstPath := filepath.Join(dst, file.Name())
		copiedFiles[dstPath] = true
		w, err := os.Create(dstPath)
		if err != nil {
			base.Fatal(err)
		}
		if _, err := io.Copy(w, r); err != nil {
			base.Fatal(err)
		}
		r.Close()
		if err := w.Close(); err != nil {
			base.Fatal(err)
		}
	}
}

// checkPathCollisions will fail if case-insensitive collisions are present.
// The reason why we do this check in go mod vendor is to keep consistency
// with go build. If modifying, consider changing load() in
// src/cmd/go/internal/load/pkg.go
func checkPathCollisions(modpkgs map[module.Version][]string) {
	var foldPath = make(map[string]string, len(modpkgs))
	for m := range modpkgs {
		fold := str.ToFold(m.Path)
		if other := foldPath[fold]; other == "" {
			foldPath[fold] = m.Path
		} else if other != m.Path {
			base.Fatalf("go.mod: case-insensitive import collision: %q and %q", m.Path, other)
		}
	}
}
