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
	UsageLine: "list [-f format] [-json] [importpath...]",
	Short:     "list packages",
	Long: `
List lists the packages named by the import paths, one per line.

The default output shows the package name and file system location:

    books /home/you/src/google-api-go-client.googlecode.com/hg/books/v1
    oauth /home/you/src/goauth2.googlecode.com/hg/oauth
    sqlite /home/you/src/gosqlite.googlecode.com/hg/sqlite

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
        GoFiles  []string // .go source files (excluding CgoFiles)
        CFiles   []string // .c source files
        HFiles   []string // .h source files
        SFiles   []string // .s source files
        CgoFiles []string // .go sources files that import "C"

        // Dependency information
        Imports []string // import paths used by this package
        Deps    []string // all (recursively) imported dependencies
    }

The -json flag causes the package data to be printed in JSON format
instead of using the template format.

For more about import paths, see 'go help importpath'.
	`,
}

func init() {
	cmdList.Run = runList // break init cycle
}

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

	for _, pkg := range packages(args) {
		do(pkg)
	}
}
