// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"encoding/json"
	"io"
	"os"
	"strings"
	"text/template"
)

var cmdList = &Command{
	UsageLine: "list [-e] [-f format] [-json] [-tags 'tag list'] [packages]",
	Short:     "list packages",
	Long: `
List lists the packages named by the import paths, one per line.

The default output shows the package import path:

    code.google.com/p/google-api-go-client/books/v1
    code.google.com/p/goauth2/oauth
    code.google.com/p/sqlite

The -f flag specifies an alternate format for the list, using the
syntax of package template.  The default output is equivalent to -f
'{{.ImportPath}}'.  One extra template function is available, "join",
which calls strings.Join. The struct being passed to the template is:

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
        GoFiles  []string       // .go source files (excluding CgoFiles, TestGoFiles, XTestGoFiles)
        CgoFiles []string       // .go sources files that import "C"
        IgnoredGoFiles []string // .go sources ignored due to build constraints
        CFiles   []string       // .c source files
        HFiles   []string       // .h source files
        SFiles   []string       // .s source files
        SysoFiles []string      // .syso object files to add to archive
        SwigFiles []string      // .swig files
        SwigCXXFiles []string   // .swigcxx files

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

The -tags flag specifies a list of build tags, like in the 'go build'
command.

For more about specifying packages, see 'go help packages'.
	`,
}

func init() {
	cmdList.Run = runList // break init cycle
	cmdList.Flag.Var(buildCompiler{}, "compiler", "")
	cmdList.Flag.Var((*stringsFlag)(&buildContext.BuildTags), "tags", "")
}

var listE = cmdList.Flag.Bool("e", false, "")
var listFmt = cmdList.Flag.String("f", "{{.ImportPath}}", "")
var listJson = cmdList.Flag.Bool("json", false, "")
var nl = []byte{'\n'}

func runList(cmd *Command, args []string) {
	out := newTrackingWriter(os.Stdout)
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
		tmpl, err := template.New("main").Funcs(template.FuncMap{"join": strings.Join}).Parse(*listFmt)
		if err != nil {
			fatalf("%s", err)
		}
		do = func(p *Package) {
			if err := tmpl.Execute(out, p); err != nil {
				out.Flush()
				fatalf("%s", err)
			}
			if out.NeedNL() {
				out.Write([]byte{'\n'})
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

// TrackingWriter tracks the last byte written on every write so
// we can avoid printing a newline if one was already written or
// if there is no output at all.
type TrackingWriter struct {
	w    *bufio.Writer
	last byte
}

func newTrackingWriter(w io.Writer) *TrackingWriter {
	return &TrackingWriter{
		w:    bufio.NewWriter(w),
		last: '\n',
	}
}

func (t *TrackingWriter) Write(p []byte) (n int, err error) {
	n, err = t.w.Write(p)
	if n > 0 {
		t.last = p[n-1]
	}
	return
}

func (t *TrackingWriter) Flush() {
	t.w.Flush()
}

func (t *TrackingWriter) NeedNL() bool {
	return t.last != '\n'
}
