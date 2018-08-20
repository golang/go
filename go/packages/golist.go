// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packages

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// A goTooOldError reports that the go command
// found by exec.LookPath is too old to use the new go list behavior.
type goTooOldError struct {
	error
}

// goListDriver uses the go list command to interpret the patterns and produce
// the build system package structure.
// See driver for more details.
func goListDriver(cfg *Config, patterns ...string) (*driverResponse, error) {
	// Determine files requested in contains patterns
	var containFiles []string
	restPatterns := make([]string, 0, len(patterns))
	for _, pattern := range patterns {
		if strings.HasPrefix(pattern, "contains:") {
			containFile := strings.TrimPrefix(pattern, "contains:")
			containFiles = append(containFiles, containFile)
		} else {
			restPatterns = append(restPatterns, pattern)
		}
	}
	containFiles = absJoin(cfg.Dir, containFiles)
	patterns = restPatterns

	// TODO(matloob): Remove the definition of listfunc and just use golistPackages once go1.12 is released.
	var listfunc driver
	listfunc = func(cfg *Config, words ...string) (*driverResponse, error) {
		response, err := golistDriverCurrent(cfg, patterns...)
		if _, ok := err.(goTooOldError); ok {
			listfunc = golistDriverFallback
			return listfunc(cfg, patterns...)
		}
		listfunc = golistDriverCurrent
		return response, err
	}

	var response *driverResponse
	var err error

	// see if we have any patterns to pass through to go list.
	if len(patterns) > 0 {
		response, err = listfunc(cfg, patterns...)
		if err != nil {
			return nil, err
		}
	} else {
		response = &driverResponse{}
	}

	// Run go list for contains: patterns.
	seenPkgs := make(map[string]*Package) // for deduplication. different containing queries could produce same packages
	if len(containFiles) > 0 {
		for _, pkg := range response.Packages {
			seenPkgs[pkg.ID] = pkg
		}
	}
	for _, f := range containFiles {
		// TODO(matloob): Do only one query per directory.
		fdir := filepath.Dir(f)
		cfg.Dir = fdir
		dirResponse, err := listfunc(cfg, ".")
		if err != nil {
			return nil, err
		}
		isRoot := make(map[string]bool, len(dirResponse.Roots))
		for _, root := range dirResponse.Roots {
			isRoot[root] = true
		}
		for _, pkg := range dirResponse.Packages {
			// Add any new packages to the main set
			// We don't bother to filter packages that will be dropped by the changes of roots,
			// that will happen anyway during graph construction outside this function.
			// Over-reporting packages is not a problem.
			if _, ok := seenPkgs[pkg.ID]; !ok {
				// it is a new package, just add it
				seenPkgs[pkg.ID] = pkg
				response.Packages = append(response.Packages, pkg)
			}
			// if the package was not a root one, it cannot have the file
			if !isRoot[pkg.ID] {
				continue
			}
			for _, pkgFile := range pkg.GoFiles {
				if filepath.Base(f) == filepath.Base(pkgFile) {
					response.Roots = append(response.Roots, pkg.ID)
					break
				}
			}
		}
	}
	return response, nil
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
	CXXFiles        []string
	MFiles          []string
	HFiles          []string
	FFiles          []string
	SFiles          []string
	SwigFiles       []string
	SwigCXXFiles    []string
	SysoFiles       []string
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

func otherFiles(p *jsonPackage) [][]string {
	return [][]string{p.CFiles, p.CXXFiles, p.MFiles, p.HFiles, p.FFiles, p.SFiles, p.SwigFiles, p.SwigCXXFiles, p.SysoFiles}
}

// golistDriverCurrent uses the "go list" command to expand the
// pattern words and return metadata for the specified packages.
// dir may be "" and env may be nil, as per os/exec.Command.
func golistDriverCurrent(cfg *Config, words ...string) (*driverResponse, error) {
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
	buf, err := golist(cfg, golistargs(cfg, words))
	if err != nil {
		return nil, err
	}
	// Decode the JSON and convert it to Package form.
	var response driverResponse
	for dec := json.NewDecoder(buf); dec.More(); {
		p := new(jsonPackage)
		if err := dec.Decode(p); err != nil {
			return nil, fmt.Errorf("JSON decoding failed: %v", err)
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
		imports := make(map[string]*Package)
		for path, id := range p.ImportMap {
			imports[path] = &Package{ID: id} // non-identity import
			delete(ids, id)
		}
		for id := range ids {
			if id == "C" {
				continue
			}

			imports[id] = &Package{ID: id} // identity import
		}
		if !p.DepOnly {
			response.Roots = append(response.Roots, id)
		}
		pkg := &Package{
			ID:              id,
			Name:            p.Name,
			PkgPath:         pkgpath,
			GoFiles:         absJoin(p.Dir, p.GoFiles, p.CgoFiles),
			CompiledGoFiles: absJoin(p.Dir, p.CompiledGoFiles),
			OtherFiles:      absJoin(p.Dir, otherFiles(p)...),
			Imports:         imports,
			ExportFile:      export,
		}
		// TODO(matloob): Temporary hack since CompiledGoFiles isn't always set.
		if len(pkg.CompiledGoFiles) == 0 {
			pkg.CompiledGoFiles = pkg.GoFiles
		}
		response.Packages = append(response.Packages, pkg)
	}

	return &response, nil
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

func golistargs(cfg *Config, words []string) []string {
	fullargs := []string{
		"list", "-e", "-json", "-compiled",
		fmt.Sprintf("-test=%t", cfg.Tests),
		fmt.Sprintf("-export=%t", usesExportData(cfg)),
		fmt.Sprintf("-deps=%t", cfg.Mode >= LoadImports),
	}
	fullargs = append(fullargs, cfg.BuildFlags...)
	fullargs = append(fullargs, "--")
	fullargs = append(fullargs, words...)
	return fullargs
}

// golist returns the JSON-encoded result of a "go list args..." query.
func golist(cfg *Config, args []string) (*bytes.Buffer, error) {
	out := new(bytes.Buffer)
	cmd := exec.CommandContext(cfg.Context, "go", args...)
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
		if !usesExportData(cfg) {
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
