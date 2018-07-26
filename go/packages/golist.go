// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packages

// This file defines the "go list" implementation of the Packages metadata query.

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

// A GoTooOldError reports that the go command
// found by exec.LookPath does not contain the necessary
// support to be used with go/packages.
// Currently, go/packages requires Go 1.11 or later.
// (We intend to issue a point release for Go 1.10
// so that go/packages can be used with updated Go 1.10 systems too.)
type GoTooOldError struct {
	error
}

// Fields must match go list;
// see $GOROOT/src/cmd/go/internal/load/pkg.go.
type jsonPackage struct {
	ImportPath   string
	Dir          string
	Name         string
	Export       string
	GoFiles      []string
	CFiles       []string
	CgoFiles     []string
	SFiles       []string
	Imports      []string
	ImportMap    map[string]string
	Deps         []string
	TestGoFiles  []string
	TestImports  []string
	XTestGoFiles []string
	XTestImports []string
	ForTest      string // q in a "p [q.test]" package, else ""
	DepOnly      bool
}

// golistPackages uses the "go list" command to expand the
// pattern words and return metadata for the specified packages.
// dir may be "" and env may be nil, as per os/exec.Command.
func golistPackages(cfg *rawConfig, words ...string) ([]*rawPackage, error) {
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
	// Decode the JSON and convert it to rawPackage form.
	var result []*rawPackage
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

		pkg := &rawPackage{
			ID:         id,
			Name:       p.Name,
			GoFiles:    absJoin(p.Dir, p.GoFiles, p.CgoFiles),
			OtherFiles: absJoin(p.Dir, p.SFiles, p.CFiles),
			PkgPath:    pkgpath,
			Imports:    imports,
			Export:     export,
			DepOnly:    p.DepOnly,
		}
		result = append(result, pkg)
	}

	return result, nil
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

func golistargs(cfg *rawConfig, words []string) []string {
	fullargs := []string{"list", "-e", "-json", "-cgo=true"}
	fullargs = append(fullargs, cfg.Flags()...)
	fullargs = append(fullargs, "--")
	fullargs = append(fullargs, words...)
	return fullargs
}

// golist returns the JSON-encoded result of a "go list args..." query.
func golist(cfg *rawConfig, args []string) (*bytes.Buffer, error) {
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
			return nil, GoTooOldError{fmt.Errorf("unsupported version of go list: %s: %s", exitErr, cmd.Stderr)}
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
