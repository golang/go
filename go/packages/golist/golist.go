// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package golist defines the "go list" implementation of the Packages metadata query.
package golist

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"golang.org/x/tools/go/packages/raw"
)

// A goTooOldError reports that the go command
// found by exec.LookPath is too old to use the new go list behavior.
type goTooOldError struct {
	error
}

// LoadRaw and returns the raw Go packages named by the given patterns.
// This is a low level API, in general you should be using the packages.Load
// unless you have a very strong need for the raw data, and you know that you
// are using conventional go layout as supported by `go list`
// It returns the packages identifiers that directly matched the patterns, the
// full set of packages requested (which may include the dependencies) and
// an error if the operation failed.
func LoadRaw(ctx context.Context, cfg *raw.Config, patterns ...string) ([]string, []*raw.Package, error) {
	if len(patterns) == 0 {
		return nil, nil, fmt.Errorf("no packages to load")
	}
	if cfg == nil {
		return nil, nil, fmt.Errorf("Load must be passed a valid Config")
	}
	if cfg.Dir == "" {
		return nil, nil, fmt.Errorf("Config does not have a working directory")
	}
	// Determine files requested in contains patterns
	var containFiles []string
	{
		restPatterns := make([]string, 0, len(patterns))
		for _, pattern := range patterns {
			if containFile := strings.TrimPrefix(pattern, "contains:"); containFile != pattern {
				containFiles = append(containFiles, containFile)
			} else {
				restPatterns = append(restPatterns, pattern)
			}
		}
		containFiles = absJoin(cfg.Dir, containFiles)
		patterns = restPatterns
	}

	// TODO(matloob): Remove the definition of listfunc and just use golistPackages once go1.12 is released.
	var listfunc func(ctx context.Context, cfg *raw.Config, words ...string) ([]string, []*raw.Package, error)
	listfunc = func(ctx context.Context, cfg *raw.Config, words ...string) ([]string, []*raw.Package, error) {
		roots, pkgs, err := golistPackages(ctx, cfg, patterns...)
		if _, ok := err.(goTooOldError); ok {
			listfunc = golistPackagesFallback
			return listfunc(ctx, cfg, patterns...)
		}
		listfunc = golistPackages
		return roots, pkgs, err
	}

	roots, pkgs, err := []string(nil), []*raw.Package(nil), error(nil)

	// TODO(matloob): Patterns may now be empty, if it was solely comprised of contains: patterns.
	// See if the extra process invocation can be avoided.
	if len(patterns) > 0 {
		roots, pkgs, err = listfunc(ctx, cfg, patterns...)
		if err != nil {
			return nil, nil, err
		}
	}

	// Run go list for contains: patterns.
	seenPkgs := make(map[string]bool) // for deduplication. different containing queries could produce same packages
	seenRoots := make(map[string]bool)
	if len(containFiles) > 0 {
		for _, pkg := range pkgs {
			seenPkgs[pkg.ID] = true
		}
	}
	for _, f := range containFiles {
		// TODO(matloob): Do only one query per directory.
		fdir := filepath.Dir(f)
		cfg.Dir = fdir
		_, cList, err := listfunc(ctx, cfg, ".")
		if err != nil {
			return nil, nil, err
		}
		// Deduplicate and set deplist to set of packages requested files.
		for _, pkg := range cList {
			if seenRoots[pkg.ID] {
				continue
			}
			for _, pkgFile := range pkg.GoFiles {
				if filepath.Base(f) == filepath.Base(pkgFile) {
					seenRoots[pkg.ID] = true
					roots = append(roots, pkg.ID)
					break
				}
			}
			if seenPkgs[pkg.ID] {
				continue
			}
			seenPkgs[pkg.ID] = true
			pkgs = append(pkgs, pkg)
		}
	}
	return roots, pkgs, nil
}

// Fields must match go list;
// see $GOROOT/src/cmd/go/internal/load/pkg.go.
type jsonPackage struct {
	ImportPath      string
	Dir             string
	Name            string
	Export          string
	GoFiles         []string
	CompiledGoFiles []string
	CFiles          []string
	CgoFiles        []string
	SFiles          []string
	Imports         []string
	ImportMap       map[string]string
	Deps            []string
	TestGoFiles     []string
	TestImports     []string
	XTestGoFiles    []string
	XTestImports    []string
	ForTest         string // q in a "p [q.test]" package, else ""
	DepOnly         bool
}

// golistPackages uses the "go list" command to expand the
// pattern words and return metadata for the specified packages.
// dir may be "" and env may be nil, as per os/exec.Command.
func golistPackages(ctx context.Context, cfg *raw.Config, words ...string) ([]string, []*raw.Package, error) {
	// go list uses the following identifiers in ImportPath and Imports:
	//
	// 	"p"			-- importable package or main (command)
	//      "q.test"		-- q's test executable
	// 	"p [q.test]"		-- variant of p as built for q's test executable
	//	"q_test [q.test]"	-- q's external test package
	//
	// The packages p that are built differently for a test q.test
	// are q itself, plus any helpers used by the external test q_test,
	// typically including "testing" and all its dependencies.

	// Run "go list" for complete
	// information on the specified packages.

	buf, err := golist(ctx, cfg, golistargs(cfg, words))
	if err != nil {
		return nil, nil, err
	}
	// Decode the JSON and convert it to Package form.
	var roots []string
	var result []*raw.Package
	for dec := json.NewDecoder(buf); dec.More(); {
		p := new(jsonPackage)
		if err := dec.Decode(p); err != nil {
			return nil, nil, fmt.Errorf("JSON decoding failed: %v", err)
		}

		// Bad package?
		if p.Name == "" {
			// This could be due to:
			// - no such package
			// - package directory contains no Go source files
			// - all package declarations are mangled
			// - and possibly other things.
			//
			// For now, we throw it away and let later
			// stages rediscover the problem, but this
			// discards the error message computed by go list
			// and computes a new one---by different logic:
			// if only one of the package declarations is
			// bad, for example, should we report an error
			// in Metadata mode?
			// Unless we parse and typecheck, we might not
			// notice there's a problem.
			//
			// Perhaps we should save a map of PackageID to
			// errors for such cases.
			continue
		}

		id := p.ImportPath

		// Extract the PkgPath from the package's ID.
		pkgpath := id
		if i := strings.IndexByte(id, ' '); i >= 0 {
			pkgpath = id[:i]
		}

		if pkgpath == "unsafe" {
			p.GoFiles = nil // ignore fake unsafe.go file
		}

		// Assume go list emits only absolute paths for Dir.
		if !filepath.IsAbs(p.Dir) {
			log.Fatalf("internal error: go list returned non-absolute Package.Dir: %s", p.Dir)
		}

		export := p.Export
		if export != "" && !filepath.IsAbs(export) {
			export = filepath.Join(p.Dir, export)
		}

		// imports
		//
		// Imports contains the IDs of all imported packages.
		// ImportsMap records (path, ID) only where they differ.
		ids := make(map[string]bool)
		for _, id := range p.Imports {
			ids[id] = true
		}
		imports := make(map[string]string)
		for path, id := range p.ImportMap {
			imports[path] = id // non-identity import
			delete(ids, id)
		}
		for id := range ids {
			// Go issue 26136: go list omits imports in cgo-generated files.
			if id == "C" {
				imports["unsafe"] = "unsafe"
				imports["syscall"] = "syscall"
				if pkgpath != "runtime/cgo" {
					imports["runtime/cgo"] = "runtime/cgo"
				}
				continue
			}

			imports[id] = id // identity import
		}

		pkg := &raw.Package{
			ID:              id,
			Name:            p.Name,
			GoFiles:         absJoin(p.Dir, p.GoFiles, p.CgoFiles),
			CompiledGoFiles: absJoin(p.Dir, p.CompiledGoFiles),
			OtherFiles:      absJoin(p.Dir, p.SFiles, p.CFiles),
			PkgPath:         pkgpath,
			Imports:         imports,
			Export:          export,
		}
		// TODO(matloob): Temporary hack since CompiledGoFiles isn't always set.
		if len(pkg.CompiledGoFiles) == 0 {
			pkg.CompiledGoFiles = pkg.GoFiles
		}
		if !p.DepOnly {
			roots = append(roots, pkg.ID)
		}
		result = append(result, pkg)
	}

	return roots, result, nil
}

// absJoin absolutizes and flattens the lists of files.
func absJoin(dir string, fileses ...[]string) (res []string) {
	for _, files := range fileses {
		for _, file := range files {
			if !filepath.IsAbs(file) {
				file = filepath.Join(dir, file)
			}
			res = append(res, file)
		}
	}
	return res
}

func golistargs(cfg *raw.Config, words []string) []string {
	fullargs := []string{
		"list", "-e", "-json", "-compiled",
		fmt.Sprintf("-test=%t", cfg.Tests),
		fmt.Sprintf("-export=%t", cfg.Export),
		fmt.Sprintf("-deps=%t", cfg.Deps),
	}
	fullargs = append(fullargs, cfg.Flags...)
	fullargs = append(fullargs, "--")
	fullargs = append(fullargs, words...)
	return fullargs
}

// golist returns the JSON-encoded result of a "go list args..." query.
func golist(ctx context.Context, cfg *raw.Config, args []string) (*bytes.Buffer, error) {
	out := new(bytes.Buffer)
	cmd := exec.CommandContext(ctx, "go", args...)
	cmd.Env = cfg.Env
	cmd.Dir = cfg.Dir
	cmd.Stdout = out
	cmd.Stderr = new(bytes.Buffer)
	if err := cmd.Run(); err != nil {
		exitErr, ok := err.(*exec.ExitError)
		if !ok {
			// Catastrophic error:
			// - executable not found
			// - context cancellation
			return nil, fmt.Errorf("couldn't exec 'go list': %s %T", err, err)
		}

		// Old go list?
		if strings.Contains(fmt.Sprint(cmd.Stderr), "flag provided but not defined") {
			return nil, goTooOldError{fmt.Errorf("unsupported version of go list: %s: %s", exitErr, cmd.Stderr)}
		}

		// Export mode entails a build.
		// If that build fails, errors appear on stderr
		// (despite the -e flag) and the Export field is blank.
		// Do not fail in that case.
		if !cfg.Export {
			return nil, fmt.Errorf("go list: %s: %s", exitErr, cmd.Stderr)
		}
	}

	// Print standard error output from "go list".
	// Due to the -e flag, this should be empty.
	// However, in -export mode it contains build errors.
	// Should go list save build errors in the Package.Error JSON field?
	// See https://github.com/golang/go/issues/26319.
	// If so, then we should continue to print stderr as go list
	// will be silent unless something unexpected happened.
	// If not, perhaps we should suppress it to reduce noise.
	if stderr := fmt.Sprint(cmd.Stderr); stderr != "" {
		fmt.Fprintf(os.Stderr, "go list stderr <<%s>>\n", stderr)
	}

	// debugging
	if false {
		fmt.Fprintln(os.Stderr, out)
	}

	return out, nil
}
