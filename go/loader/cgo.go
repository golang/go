// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package loader

// This file handles cgo preprocessing of files containing `import "C"`.
//
// DESIGN
//
// The approach taken is to run the cgo processor on the package's
// CgoFiles and parse the output, faking the filenames of the
// resulting ASTs so that the synthetic file containing the C types is
// called "C" (e.g. "~/go/src/net/C") and the preprocessed files
// have their original names (e.g. "~/go/src/net/cgo_unix.go"),
// not the names of the actual temporary files.
//
// The advantage of this approach is its fidelity to 'go build'.  The
// downside is that the token.Position.Offset for each AST node is
// incorrect, being an offset within the temporary file.  Line numbers
// should still be correct because of the //line comments.
//
// The logic of this file is mostly plundered from the 'go build'
// tool, which also invokes the cgo preprocessor.
//
//
// REJECTED ALTERNATIVE
//
// An alternative approach that we explored is to extend go/types'
// Importer mechanism to provide the identity of the importing package
// so that each time `import "C"` appears it resolves to a different
// synthetic package containing just the objects needed in that case.
// The loader would invoke cgo but parse only the cgo_types.go file
// defining the package-level objects, discarding the other files
// resulting from preprocessing.
//
// The benefit of this approach would have been that source-level
// syntax information would correspond exactly to the original cgo
// file, with no preprocessing involved, making source tools like
// godoc, guru, and eg happy.  However, the approach was rejected
// due to the additional complexity it would impose on go/types.  (It
// made for a beautiful demo, though.)
//
// cgo files, despite their *.go extension, are not legal Go source
// files per the specification since they may refer to unexported
// members of package "C" such as C.int.  Also, a function such as
// C.getpwent has in effect two types, one matching its C type and one
// which additionally returns (errno C.int).  The cgo preprocessor
// uses name mangling to distinguish these two functions in the
// processed code, but go/types would need to duplicate this logic in
// its handling of function calls, analogous to the treatment of map
// lookups in which y=m[k] and y,ok=m[k] are both legal.

import (
	"fmt"
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
)

// processCgoFiles invokes the cgo preprocessor on bp.CgoFiles, parses
// the output and returns the resulting ASTs.
//
func processCgoFiles(bp *build.Package, fset *token.FileSet, DisplayPath func(path string) string, mode parser.Mode) ([]*ast.File, error) {
	tmpdir, err := ioutil.TempDir("", strings.Replace(bp.ImportPath, "/", "_", -1)+"_C")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(tmpdir)

	pkgdir := bp.Dir
	if DisplayPath != nil {
		pkgdir = DisplayPath(pkgdir)
	}

	cgoFiles, cgoDisplayFiles, err := runCgo(bp, pkgdir, tmpdir)
	if err != nil {
		return nil, err
	}
	var files []*ast.File
	for i := range cgoFiles {
		rd, err := os.Open(cgoFiles[i])
		if err != nil {
			return nil, err
		}
		display := filepath.Join(bp.Dir, cgoDisplayFiles[i])
		f, err := parser.ParseFile(fset, display, rd, mode)
		rd.Close()
		if err != nil {
			return nil, err
		}
		files = append(files, f)
	}
	return files, nil
}

var cgoRe = regexp.MustCompile(`[/\\:]`)

// runCgo invokes the cgo preprocessor on bp.CgoFiles and returns two
// lists of files: the resulting processed files (in temporary
// directory tmpdir) and the corresponding names of the unprocessed files.
//
// runCgo is adapted from (*builder).cgo in
// $GOROOT/src/cmd/go/build.go, but these features are unsupported:
// Objective C, CGOPKGPATH, CGO_FLAGS.
//
func runCgo(bp *build.Package, pkgdir, tmpdir string) (files, displayFiles []string, err error) {
	cgoCPPFLAGS, _, _, _ := cflags(bp, true)
	_, cgoexeCFLAGS, _, _ := cflags(bp, false)

	if len(bp.CgoPkgConfig) > 0 {
		pcCFLAGS, err := pkgConfigFlags(bp)
		if err != nil {
			return nil, nil, err
		}
		cgoCPPFLAGS = append(cgoCPPFLAGS, pcCFLAGS...)
	}

	// Allows including _cgo_export.h from .[ch] files in the package.
	cgoCPPFLAGS = append(cgoCPPFLAGS, "-I", tmpdir)

	// _cgo_gotypes.go (displayed "C") contains the type definitions.
	files = append(files, filepath.Join(tmpdir, "_cgo_gotypes.go"))
	displayFiles = append(displayFiles, "C")
	for _, fn := range bp.CgoFiles {
		// "foo.cgo1.go" (displayed "foo.go") is the processed Go source.
		f := cgoRe.ReplaceAllString(fn[:len(fn)-len("go")], "_")
		files = append(files, filepath.Join(tmpdir, f+"cgo1.go"))
		displayFiles = append(displayFiles, fn)
	}

	var cgoflags []string
	if bp.Goroot && bp.ImportPath == "runtime/cgo" {
		cgoflags = append(cgoflags, "-import_runtime_cgo=false")
	}
	if bp.Goroot && bp.ImportPath == "runtime/race" || bp.ImportPath == "runtime/cgo" {
		cgoflags = append(cgoflags, "-import_syscall=false")
	}

	args := stringList(
		"go", "tool", "cgo", "-objdir", tmpdir, cgoflags, "--",
		cgoCPPFLAGS, cgoexeCFLAGS, bp.CgoFiles,
	)
	if false {
		log.Printf("Running cgo for package %q: %s (dir=%s)", bp.ImportPath, args, pkgdir)
	}
	cmd := exec.Command(args[0], args[1:]...)
	cmd.Dir = pkgdir
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return nil, nil, fmt.Errorf("cgo failed: %s: %s", args, err)
	}

	return files, displayFiles, nil
}

// -- unmodified from 'go build' ---------------------------------------

// Return the flags to use when invoking the C or C++ compilers, or cgo.
func cflags(p *build.Package, def bool) (cppflags, cflags, cxxflags, ldflags []string) {
	var defaults string
	if def {
		defaults = "-g -O2"
	}

	cppflags = stringList(envList("CGO_CPPFLAGS", ""), p.CgoCPPFLAGS)
	cflags = stringList(envList("CGO_CFLAGS", defaults), p.CgoCFLAGS)
	cxxflags = stringList(envList("CGO_CXXFLAGS", defaults), p.CgoCXXFLAGS)
	ldflags = stringList(envList("CGO_LDFLAGS", defaults), p.CgoLDFLAGS)
	return
}

// envList returns the value of the given environment variable broken
// into fields, using the default value when the variable is empty.
func envList(key, def string) []string {
	v := os.Getenv(key)
	if v == "" {
		v = def
	}
	return strings.Fields(v)
}

// stringList's arguments should be a sequence of string or []string values.
// stringList flattens them into a single []string.
func stringList(args ...interface{}) []string {
	var x []string
	for _, arg := range args {
		switch arg := arg.(type) {
		case []string:
			x = append(x, arg...)
		case string:
			x = append(x, arg)
		default:
			panic("stringList: invalid argument")
		}
	}
	return x
}
