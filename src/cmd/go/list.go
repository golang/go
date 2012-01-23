// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/json"
	"os"
	"text/template"
)

var cmdList = &Command{
	UsageLine: "list [-e] [-f format] [-json] [importpath...]",
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
        Name       string // package name
        Doc        string // package documentation string
        ImportPath string // import path of package in dir
        Dir        string // directory containing package sources
        Version    string // version of installed package (TODO)
        Stale      bool   // would 'go install' do anything for this package?

        // Source files
        GoFiles      []string // .go source files (excluding CgoFiles, TestGoFiles, and XTestGoFiles)
        TestGoFiles  []string // _test.go source files internal to the package they are testing
        XTestGoFiles []string // _test.go source files external to the package they are testing
        CFiles       []string // .c source files
        HFiles       []string // .h source files
        SFiles       []string // .s source files
        CgoFiles     []string // .go sources files that import "C"

        // Dependency information
        Imports []string // import paths used by this package
        Deps    []string // all (recursively) imported dependencies
        
        // Error information
        Incomplete bool            // this package or a dependency has an error
        Error *PackageError        // error loading package
        DepsErrors []*PackageError // errors loading dependencies
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

For more about import paths, see 'go help importpath'.
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
	var do func(*Package)
	if *listJson {
		do = func(p *Package) {
			b, err := json.MarshalIndent(p, "", "\t")
			if err != nil {
				fatalf("%s", err)
			}
			os.Stdout.Write(b)
			os.Stdout.Write(nl)
		}
	} else {
		tmpl, err := template.New("main").Parse(*listFmt + "\n")
		if err != nil {
			fatalf("%s", err)
		}
		do = func(p *Package) {
			if err := tmpl.Execute(os.Stdout, p); err != nil {
				fatalf("%s", err)
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
