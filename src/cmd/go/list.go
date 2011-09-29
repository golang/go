// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var cmdList = &Command{
	Run:       runList,
	UsageLine: "list [-f format] [-json] [importpath...]",
	Short:     "list packages",
	Long: `
List lists the packages named by the import paths.

The default output shows the package name and file system location:

    books /home/you/src/google-api-go-client.googlecode.com/hg/books/v1
    oauth /home/you/src/goauth2.googlecode.com/hg/oauth
    sqlite /home/you/src/gosqlite.googlecode.com/hg/sqlite

The -f flag specifies an alternate format for the list,
using the syntax of package template.  The default output
is equivalent to -f '{{.Name}} {{.Dir}}'  The struct
being passed to the template is:

    type Package struct {
        Name string         // package name
        Doc string          // package documentation string
        GoFiles []string    // names of Go source files in package
        ImportPath string   // import path denoting package
        Imports []string    // import paths used by this package
        Deps []string       // all (recursively) imported dependencies
        Dir string          // directory containing package sources
        Version string      // version of installed package
    }

The -json flag causes the package data to be printed in JSON format.

For more about import paths, see 'go help importpath'.
	`,
}

var listFmt = cmdList.Flag.String("f", "{{.Name}} {{.Dir}}", "")
var listJson = cmdList.Flag.Bool("json", false, "")

func runList(cmd *Command, args []string) {
	args = importPaths(args)
	_ = args
	panic("list not implemented")
}
