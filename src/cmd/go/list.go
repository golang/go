// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"encoding/json"
	"io"
	"os"
	"text/template"
)

var cmdList = &Command{
	UsageLine: "list [-e] [-f format] [-json] [packages]",
	Short:     "list packages",
	Long: `
List lists the packages named by the import paths, one per line.

The default output shows the package import path:

    code.google.com/p/google-api-go-client/books/v1
    code.google.com/p/goauth2/oauth
    code.google.com/p/sqlite

The -f flag specifies an alternate format for the list,
using the syntax of package template.  The default output
is equivalent to -f '{{.ImportPath}}'.  The struct
being passed to the template is:

    type Package struct {
        Dir        string // directory containing package sources
        ImportPath string // import path of package in dir
        Name       string // package name
        Doc        string // package documentation string
        Target     string // install path
        Goroot     bool   // is this package in the Go root?
        Standard   bool   // is this package part of the standard Go library?
        Stale      bool   // would 'go install' do anything for this package?
        Root       string // Go root or Go path dir containing this package

        // Source files
        GoFiles  []string // .go source files (excluding CgoFiles, TestGoFiles, XTestGoFiles)
        CgoFiles []string // .go sources files that import "C"
        CFiles   []string // .c source files
        HFiles   []string // .h source files
        SFiles   []string // .s source files

        // Cgo directives
        CgoCFLAGS    []string // cgo: flags for C compiler
        CgoLDFLAGS   []string // cgo: flags for linker
        CgoPkgConfig []string // cgo: pkg-config names

        // Dependency information
        Imports []string // import paths used by this package
        Deps    []string // all (recursively) imported dependencies

        // Error information
        Incomplete bool            // this package or a dependency has an error
        Error      *PackageError   // error loading package
        DepsErrors []*PackageError // errors loading dependencies

        TestGoFiles  []string // _test.go files in package
        TestImports  []string // imports from TestGoFiles
        XTestGoFiles []string // _test.go files outside package
        XTestImports []string // imports from XTestGoFiles
    }

The -json flag causes the package data to be printed in JSON format
instead of using the template format.

The -e flag changes the handling of erroneous packages, those that
cannot be found or are malformed.  By default, the list command
prints an error to standard error for each erroneous package and
omits the packages from consideration during the usual printing.
With the -e flag, the list command never prints errors to standard
error and instead processes the erroneous packages with the usual
printing.  Erroneous packages will have a non-empty ImportPath and
a non-nil Error field; other information may or may not be missing
(zeroed).

For more about specifying packages, see 'go help packages'.
	`,
}

func init() {
	cmdList.Run = runList // break init cycle
}

var listE = cmdList.Flag.Bool("e", false, "")
var listFmt = cmdList.Flag.String("f", "{{.ImportPath}}", "")
var listJson = cmdList.Flag.Bool("json", false, "")
var nl = []byte{'\n'}

func runList(cmd *Command, args []string) {
	out := newCountingWriter(os.Stdout)
	defer out.w.Flush()

	var do func(*Package)
	if *listJson {
		do = func(p *Package) {
			b, err := json.MarshalIndent(p, "", "\t")
			if err != nil {
				out.Flush()
				fatalf("%s", err)
			}
			out.Write(b)
			out.Write(nl)
		}
	} else {
		tmpl, err := template.New("main").Parse(*listFmt)
		if err != nil {
			fatalf("%s", err)
		}
		do = func(p *Package) {
			out.Reset()
			if err := tmpl.Execute(out, p); err != nil {
				out.Flush()
				fatalf("%s", err)
			}
			if out.Count() > 0 {
				out.w.WriteRune('\n')
			}
		}
	}

	load := packages
	if *listE {
		load = packagesAndErrors
	}

	for _, pkg := range load(args) {
		do(pkg)
	}
}

// CountingWriter counts its data, so we can avoid appending a newline
// if there was no actual output.
type CountingWriter struct {
	w     *bufio.Writer
	count int64
}

func newCountingWriter(w io.Writer) *CountingWriter {
	return &CountingWriter{
		w: bufio.NewWriter(w),
	}
}

func (cw *CountingWriter) Write(p []byte) (n int, err error) {
	cw.count += int64(len(p))
	return cw.w.Write(p)
}

func (cw *CountingWriter) Flush() {
	cw.w.Flush()
}

func (cw *CountingWriter) Reset() {
	cw.count = 0
}

func (cw *CountingWriter) Count() int64 {
	return cw.count
}
