// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package rename contains the implementation of the 'gorename' command
// whose main function is in golang.org/x/tools/refactor/rename.
// See that package for the command documentation.
package rename

import (
	"errors"
	"fmt"
	"go/ast"
	"go/build"
	"go/format"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"golang.org/x/tools/go/loader"
	"golang.org/x/tools/go/types"
	"golang.org/x/tools/refactor/importgraph"
	"golang.org/x/tools/refactor/satisfy"
)

var (
	// Force enables patching of the source files even if conflicts were reported.
	// The resulting program may be ill-formed.
	// It may even cause gorename to crash.  TODO(adonovan): fix that.
	Force bool

	// DryRun causes the tool to report conflicts but not update any files.
	DryRun bool

	// ConflictError is returned by Main when it aborts the renaming due to conflicts.
	// (It is distinguished because the interesting errors are the conflicts themselves.)
	ConflictError = errors.New("renaming aborted due to conflicts")

	// Verbose enables extra logging.
	Verbose bool
)

type renamer struct {
	iprog              *loader.Program
	objsToUpdate       map[types.Object]bool
	hadConflicts       bool
	to                 string
	satisfyConstraints map[satisfy.Constraint]bool
	packages           map[*types.Package]*loader.PackageInfo // subset of iprog.AllPackages to inspect
}

var reportError = func(posn token.Position, message string) {
	fmt.Fprintf(os.Stderr, "%s: %s\n", posn, message)
}

func Main(ctxt *build.Context, offsetFlag, fromFlag, to string) error {
	// -- Parse the -from or -offset specifier ----------------------------

	if (offsetFlag == "") == (fromFlag == "") {
		return fmt.Errorf("exactly one of the -from and -offset flags must be specified")
	}

	if !isValidIdentifier(to) {
		return fmt.Errorf("-to %q: not a valid identifier", to)
	}

	var spec *spec
	var err error
	if fromFlag != "" {
		spec, err = parseFromFlag(ctxt, fromFlag)
	} else {
		spec, err = parseOffsetFlag(ctxt, offsetFlag)
	}
	if err != nil {
		return err
	}

	if spec.fromName == to {
		return fmt.Errorf("the old and new names are the same: %s", to)
	}

	// -- Load the program consisting of the initial package  -------------

	iprog, err := loadProgram(ctxt, map[string]bool{spec.pkg: true})
	if err != nil {
		return err
	}

	fromObjects, err := findFromObjects(iprog, spec)
	if err != nil {
		return err
	}

	// -- Load a larger program, for global renamings ---------------------

	if requiresGlobalRename(fromObjects, to) {
		// For a local refactoring, we needn't load more
		// packages, but if the renaming affects the package's
		// API, we we must load all packages that depend on the
		// package defining the object, plus their tests.

		if Verbose {
			fmt.Fprintln(os.Stderr, "Potentially global renaming; scanning workspace...")
		}

		// Scan the workspace and build the import graph.
		_, rev, errors := importgraph.Build(ctxt)
		if len(errors) > 0 {
			fmt.Fprintf(os.Stderr, "While scanning Go workspace:\n")
			for path, err := range errors {
				fmt.Fprintf(os.Stderr, "Package %q: %s.\n", path, err)
			}
		}

		// Enumerate the set of potentially affected packages.
		affectedPackages := make(map[string]bool)
		for _, obj := range fromObjects {
			// External test packages are never imported,
			// so they will never appear in the graph.
			for path := range rev.Search(obj.Pkg().Path()) {
				affectedPackages[path] = true
			}
		}

		// TODO(adonovan): allow the user to specify the scope,
		// or -ignore patterns?  Computing the scope when we
		// don't (yet) support inputs containing errors can make
		// the tool rather brittle.

		// Re-load the larger program.
		iprog, err = loadProgram(ctxt, affectedPackages)
		if err != nil {
			return err
		}

		fromObjects, err = findFromObjects(iprog, spec)
		if err != nil {
			return err
		}
	}

	// -- Do the renaming -------------------------------------------------

	r := renamer{
		iprog:        iprog,
		objsToUpdate: make(map[types.Object]bool),
		to:           to,
		packages:     make(map[*types.Package]*loader.PackageInfo),
	}

	// Only the initially imported packages (iprog.Imported) and
	// their external tests (iprog.Created) should be inspected or
	// modified, as only they have type-checked functions bodies.
	// The rest are just dependencies, needed only for package-level
	// type information.
	for _, info := range iprog.Imported {
		r.packages[info.Pkg] = info
	}
	for _, info := range iprog.Created { // (tests)
		r.packages[info.Pkg] = info
	}

	for _, from := range fromObjects {
		r.check(from)
	}
	if r.hadConflicts && !Force {
		return ConflictError
	}
	if DryRun {
		// TODO(adonovan): print the delta?
		return nil
	}
	return r.update()
}

// loadProgram loads the specified set of packages (plus their tests)
// and all their dependencies, from source, through the specified build
// context.  Only packages in pkgs will have their functions bodies typechecked.
func loadProgram(ctxt *build.Context, pkgs map[string]bool) (*loader.Program, error) {
	conf := loader.Config{
		Build:         ctxt,
		SourceImports: true,
		ParserMode:    parser.ParseComments,

		// TODO(adonovan): enable this.  Requires making a lot of code more robust!
		AllowErrors: false,
	}

	// Optimization: don't type-check the bodies of functions in our
	// dependencies, since we only need exported package members.
	conf.TypeCheckFuncBodies = func(p string) bool {
		return pkgs[p] || pkgs[strings.TrimSuffix(p, "_test")]
	}

	if Verbose {
		var list []string
		for pkg := range pkgs {
			list = append(list, pkg)
		}
		sort.Strings(list)
		for _, pkg := range list {
			fmt.Fprintf(os.Stderr, "Loading package: %s\n", pkg)
		}
	}

	for pkg := range pkgs {
		if err := conf.ImportWithTests(pkg); err != nil {
			return nil, err
		}
	}
	return conf.Load()
}

// requiresGlobalRename reports whether this renaming could potentially
// affect other packages in the Go workspace.
func requiresGlobalRename(fromObjects []types.Object, to string) bool {
	var tfm bool
	for _, from := range fromObjects {
		if from.Exported() {
			return true
		}
		switch objectKind(from) {
		case "type", "field", "method":
			tfm = true
		}
	}
	if ast.IsExported(to) && tfm {
		// A global renaming may be necessary even if we're
		// exporting a previous unexported name, since if it's
		// the name of a type, field or method, this could
		// change selections in other packages.
		// (We include "type" in this list because a type
		// used as an embedded struct field entails a field
		// renaming.)
		return true
	}
	return false
}

// update updates the input files.
func (r *renamer) update() error {
	// We use token.File, not filename, since a file may appear to
	// belong to multiple packages and be parsed more than once.
	// token.File captures this distinction; filename does not.
	var nidents int
	var filesToUpdate = make(map[*token.File]bool)
	for _, info := range r.packages {
		// Mutate the ASTs and note the filenames.
		for id, obj := range info.Defs {
			if r.objsToUpdate[obj] {
				nidents++
				id.Name = r.to
				filesToUpdate[r.iprog.Fset.File(id.Pos())] = true
			}
		}
		for id, obj := range info.Uses {
			if r.objsToUpdate[obj] {
				nidents++
				id.Name = r.to
				filesToUpdate[r.iprog.Fset.File(id.Pos())] = true
			}
		}
	}

	// TODO(adonovan): don't rewrite cgo + generated files.
	var nerrs, npkgs int
	for _, info := range r.packages {
		first := true
		for _, f := range info.Files {
			tokenFile := r.iprog.Fset.File(f.Pos())
			if filesToUpdate[tokenFile] {
				if first {
					npkgs++
					first = false
					if Verbose {
						fmt.Fprintf(os.Stderr, "Updating package %s\n",
							info.Pkg.Path())
					}
				}
				if err := rewriteFile(r.iprog.Fset, f, tokenFile.Name()); err != nil {
					fmt.Fprintf(os.Stderr, "gorename: %s\n", err)
					nerrs++
				}
			}
		}
	}
	fmt.Fprintf(os.Stderr, "Renamed %d occurrence%s in %d file%s in %d package%s.\n",
		nidents, plural(nidents),
		len(filesToUpdate), plural(len(filesToUpdate)),
		npkgs, plural(npkgs))
	if nerrs > 0 {
		return fmt.Errorf("failed to rewrite %d file%s", nerrs, plural(nerrs))
	}
	return nil
}

func plural(n int) string {
	if n != 1 {
		return "s"
	}
	return ""
}

func writeFile(name string, fset *token.FileSet, f *ast.File) error {
	out, err := os.Create(name)
	if err != nil {
		// assume error includes the filename
		return fmt.Errorf("failed to open file: %s", err)
	}
	if err := format.Node(out, fset, f); err != nil {
		out.Close() // ignore error
		return fmt.Errorf("failed to write file: %s", err)
	}
	return out.Close()
}

var rewriteFile = func(fset *token.FileSet, f *ast.File, orig string) (err error) {
	backup := orig + ".gorename.backup"
	// TODO(adonovan): print packages and filenames in a form useful
	// to editors (so they can reload files).
	if Verbose {
		fmt.Fprintf(os.Stderr, "\t%s\n", orig)
	}
	if err := os.Rename(orig, backup); err != nil {
		return fmt.Errorf("failed to make backup %s -> %s: %s",
			orig, filepath.Base(backup), err)
	}
	if err := writeFile(orig, fset, f); err != nil {
		// Restore the file from the backup.
		os.Remove(orig)         // ignore error
		os.Rename(backup, orig) // ignore error
		return err
	}
	os.Remove(backup) // ignore error
	return nil
}
