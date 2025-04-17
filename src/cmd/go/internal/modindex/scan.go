// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modindex

import (
	"cmd/go/internal/base"
	"cmd/go/internal/fsys"
	"cmd/go/internal/str"
	"encoding/json"
	"errors"
	"fmt"
	"go/build"
	"go/doc"
	"go/scanner"
	"go/token"
	"io/fs"
	"path/filepath"
	"strings"
)

// moduleWalkErr returns filepath.SkipDir if the directory isn't relevant
// when indexing a module or generating a filehash, ErrNotIndexed,
// if the module shouldn't be indexed, and nil otherwise.
func moduleWalkErr(root string, path string, d fs.DirEntry, err error) error {
	if err != nil {
		return ErrNotIndexed
	}
	// stop at module boundaries
	if d.IsDir() && path != root {
		if info, err := fsys.Stat(filepath.Join(path, "go.mod")); err == nil && !info.IsDir() {
			return filepath.SkipDir
		}
	}
	if d.Type()&fs.ModeSymlink != 0 {
		if target, err := fsys.Stat(path); err == nil && target.IsDir() {
			// return an error to make the module hash invalid.
			// Symlink directories in modules are tricky, so we won't index
			// modules that contain them.
			// TODO(matloob): perhaps don't return this error if the symlink leads to
			// a directory with a go.mod file.
			return ErrNotIndexed
		}
	}
	return nil
}

// indexModule indexes the module at the given directory and returns its
// encoded representation. It returns ErrNotIndexed if the module can't
// be indexed because it contains symlinks.
func indexModule(modroot string) ([]byte, error) {
	fsys.Trace("indexModule", modroot)
	var packages []*rawPackage

	// If the root itself is a symlink to a directory,
	// we want to follow it (see https://go.dev/issue/50807).
	// Add a trailing separator to force that to happen.
	root := str.WithFilePathSeparator(modroot)
	err := fsys.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err := moduleWalkErr(root, path, d, err); err != nil {
			return err
		}

		if !d.IsDir() {
			return nil
		}
		if !strings.HasPrefix(path, root) {
			panic(fmt.Errorf("path %v in walk doesn't have modroot %v as prefix", path, modroot))
		}
		rel := path[len(root):]
		packages = append(packages, importRaw(modroot, rel))
		return nil
	})
	if err != nil {
		return nil, err
	}
	return encodeModuleBytes(packages), nil
}

// indexPackage indexes the package at the given directory and returns its
// encoded representation. It returns ErrNotIndexed if the package can't
// be indexed.
func indexPackage(modroot, pkgdir string) []byte {
	fsys.Trace("indexPackage", pkgdir)
	p := importRaw(modroot, relPath(pkgdir, modroot))
	return encodePackageBytes(p)
}

// rawPackage holds the information from each package that's needed to
// fill a build.Package once the context is available.
type rawPackage struct {
	error string
	dir   string // directory containing package sources, relative to the module root

	// Source files
	sourceFiles []*rawFile
}

type parseError struct {
	ErrorList   *scanner.ErrorList
	ErrorString string
}

// parseErrorToString converts the error from parsing the file into a string
// representation. A nil error is converted to an empty string, and all other
// errors are converted to a JSON-marshaled parseError struct, with ErrorList
// set for errors of type scanner.ErrorList, and ErrorString set to the error's
// string representation for all other errors.
func parseErrorToString(err error) string {
	if err == nil {
		return ""
	}
	var p parseError
	if e, ok := err.(scanner.ErrorList); ok {
		p.ErrorList = &e
	} else {
		p.ErrorString = e.Error()
	}
	s, err := json.Marshal(p)
	if err != nil {
		panic(err) // This should be impossible because scanner.Error contains only strings and ints.
	}
	return string(s)
}

// parseErrorFromString converts a string produced by parseErrorToString back
// to an error.  An empty string is converted to a nil error, and all
// other strings are expected to be JSON-marshaled parseError structs.
// The two functions are meant to preserve the structure of an
// error of type scanner.ErrorList in a round trip, but may not preserve the
// structure of other errors.
func parseErrorFromString(s string) error {
	if s == "" {
		return nil
	}
	var p parseError
	if err := json.Unmarshal([]byte(s), &p); err != nil {
		base.Fatalf(`go: invalid parse error value in index: %q. This indicates a corrupted index. Run "go clean -cache" to reset the module cache.`, s)
	}
	if p.ErrorList != nil {
		return *p.ErrorList
	}
	return errors.New(p.ErrorString)
}

// rawFile is the struct representation of the file holding all
// information in its fields.
type rawFile struct {
	error      string
	parseError string

	name                 string
	synopsis             string // doc.Synopsis of package comment... Compute synopsis on all of these?
	pkgName              string
	ignoreFile           bool   // starts with _ or . or should otherwise always be ignored
	binaryOnly           bool   // cannot be rebuilt from source (has //go:binary-only-package comment)
	cgoDirectives        string // the #cgo directive lines in the comment on import "C"
	goBuildConstraint    string
	plusBuildConstraints []string
	imports              []rawImport
	embeds               []embed
	directives           []build.Directive
}

type rawImport struct {
	path     string
	position token.Position
}

type embed struct {
	pattern  string
	position token.Position
}

// importRaw fills the rawPackage from the package files in srcDir.
// dir is the package's path relative to the modroot.
func importRaw(modroot, reldir string) *rawPackage {
	p := &rawPackage{
		dir: reldir,
	}

	absdir := filepath.Join(modroot, reldir)

	// We still haven't checked
	// that p.dir directory exists. This is the right time to do that check.
	// We can't do it earlier, because we want to gather partial information for the
	// non-nil *build.Package returned when an error occurs.
	// We need to do this before we return early on FindOnly flag.
	if !isDir(absdir) {
		// package was not found
		p.error = fmt.Errorf("cannot find package in:\n\t%s", absdir).Error()
		return p
	}

	entries, err := fsys.ReadDir(absdir)
	if err != nil {
		p.error = err.Error()
		return p
	}

	fset := token.NewFileSet()
	for _, d := range entries {
		if d.IsDir() {
			continue
		}
		if d.Type()&fs.ModeSymlink != 0 {
			if isDir(filepath.Join(absdir, d.Name())) {
				// Symlinks to directories are not source files.
				continue
			}
		}

		name := d.Name()
		ext := nameExt(name)

		if strings.HasPrefix(name, "_") || strings.HasPrefix(name, ".") {
			continue
		}
		info, err := getFileInfo(absdir, name, fset)
		if err == errNonSource {
			// not a source or object file. completely ignore in the index
			continue
		} else if err != nil {
			p.sourceFiles = append(p.sourceFiles, &rawFile{name: name, error: err.Error()})
			continue
		} else if info == nil {
			p.sourceFiles = append(p.sourceFiles, &rawFile{name: name, ignoreFile: true})
			continue
		}
		rf := &rawFile{
			name:                 name,
			goBuildConstraint:    info.goBuildConstraint,
			plusBuildConstraints: info.plusBuildConstraints,
			binaryOnly:           info.binaryOnly,
			directives:           info.directives,
		}
		if info.parsed != nil {
			rf.pkgName = info.parsed.Name.Name
		}

		// Going to save the file. For non-Go files, can stop here.
		p.sourceFiles = append(p.sourceFiles, rf)
		if ext != ".go" {
			continue
		}

		if info.parseErr != nil {
			rf.parseError = parseErrorToString(info.parseErr)
			// Fall through: we might still have a partial AST in info.Parsed,
			// and we want to list files with parse errors anyway.
		}

		if info.parsed != nil && info.parsed.Doc != nil {
			rf.synopsis = doc.Synopsis(info.parsed.Doc.Text())
		}

		var cgoDirectives []string
		for _, imp := range info.imports {
			if imp.path == "C" {
				cgoDirectives = append(cgoDirectives, extractCgoDirectives(imp.doc.Text())...)
			}
			rf.imports = append(rf.imports, rawImport{path: imp.path, position: fset.Position(imp.pos)})
		}
		rf.cgoDirectives = strings.Join(cgoDirectives, "\n")
		for _, emb := range info.embeds {
			rf.embeds = append(rf.embeds, embed{emb.pattern, emb.pos})
		}

	}
	return p
}

// extractCgoDirectives filters only the lines containing #cgo directives from the input,
// which is the comment on import "C".
func extractCgoDirectives(doc string) []string {
	var out []string
	for _, line := range strings.Split(doc, "\n") {
		// Line is
		//	#cgo [GOOS/GOARCH...] LDFLAGS: stuff
		//
		line = strings.TrimSpace(line)
		if len(line) < 5 || line[:4] != "#cgo" || (line[4] != ' ' && line[4] != '\t') {
			continue
		}

		out = append(out, line)
	}
	return out
}
