// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// licence that can be found in the LICENSE file.

// The gomvpkg command moves go packages, updating import declarations.
// See the -help message or Usage constant for details.
package main

import (
	"flag"
	"fmt"
	"go/build"
	"os"

	"golang.org/x/tools/go/buildutil"
	"golang.org/x/tools/refactor/rename"
)

var (
	fromFlag     = flag.String("from", "", "Import path of package to be moved")
	toFlag       = flag.String("to", "", "Destination import path for package")
	vcsMvCmdFlag = flag.String("vcs_mv_cmd", "", `A template for the version control system's "move directory" command, e.g. "git mv {{.Src}} {{.Dst}}`)
	helpFlag     = flag.Bool("help", false, "show usage message")
)

func init() {
	flag.Var((*buildutil.TagsFlag)(&build.Default.BuildTags), "tags", buildutil.TagsFlagDoc)
}

const Usage = `gomvpkg: moves a package, updating import declarations

Usage:

 gomvpkg -from <path> -to <path> [-vcs_mv_cmd <template>]

Flags:

-from        specifies the import path of the package to be moved

-to          specifies the destination import path

-vcs_mv_cmd  specifies a shell command to inform the version control system of a
             directory move.  The argument is a template using the syntax of the
             text/template package. It has two fields: Src and Dst, the absolute
             paths of the directories.

             For example: "git mv {{.Src}} {{.Dst}}"

gomvpkg determines the set of packages that might be affected, including all
packages importing the 'from' package and any of its subpackages. It will move
the 'from' package and all its subpackages to the destination path and update all
imports of those packages to point to its new import path.

gomvpkg rejects moves in which a package already exists at the destination import
path, or in which a directory already exists at the location the package would be
moved to.

gomvpkg will not always be able to rename imports when a package's name is changed.
Import statements may want further cleanup.

gomvpkg's behavior is not defined if any of the packages to be moved are
imported using dot imports.

Examples:

% gomvpkg -from myproject/foo -to myproject/bar

  Move the package with import path "myproject/foo" to the new path
  "myproject/bar".

% gomvpkg -from myproject/foo -to myproject/bar -vcs_mv_cmd "git mv {{.Src}} {{.Dst}}"

  Move the package with import path "myproject/foo" to the new path
  "myproject/bar" using "git mv" to execute the directory move.
`

func main() {
	flag.Parse()

	if len(flag.Args()) > 0 {
		fmt.Fprintln(os.Stderr, "gomvpkg: surplus arguments.")
		os.Exit(1)
	}

	if *helpFlag || *fromFlag == "" || *toFlag == "" {
		fmt.Println(Usage)
		return
	}

	if err := rename.Move(&build.Default, *fromFlag, *toFlag, *vcsMvCmdFlag); err != nil {
		fmt.Fprintf(os.Stderr, "gomvpkg: %s.\n", err)
		os.Exit(1)
	}
}
