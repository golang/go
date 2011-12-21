// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var helpImportpath = &Command{
	UsageLine: "importpath",
	Short:     "description of import paths",
	Long: `
Many commands apply to a set of packages named by import paths:

	go action [importpath...]

An import path that is a rooted path or that begins with
a . or .. element is interpreted as a file system path and
denotes the package in that directory.

Otherwise, the import path P denotes the package found in
the directory DIR/src/P for some DIR listed in the GOPATH
environment variable (see 'go help gopath'). 

If no import paths are given, the action applies to the
package in the current directory.

The special import path "all" expands to all package directories
found in all the GOPATH trees.  For example, 'go list all' 
lists all the packages on the local system.

The special import path "std" is like all but expands to just the
packages in the standard Go library.

An import path can also name a package to be downloaded from
a remote repository.  Run 'go help remote' for details.

Every package in a program must have a unique import path.
By convention, this is arranged by starting each path with a
unique prefix that belongs to you.  For example, paths used
internally at Google all begin with 'google', and paths
denoting remote repositories begin with the path to the code,
such as 'project.googlecode.com/'.
	`,
}

var helpRemote = &Command{
	UsageLine: "remote",
	Short:     "remote import path syntax",
	Long: `

An import path (see 'go help importpath') denotes a package
stored in the local file system.  Certain import paths also
describe how to obtain the source code for the package using
a revision control system.

A few common code hosting sites have special syntax:

	BitBucket (Mercurial)

		import "bitbucket.org/user/project"
		import "bitbucket.org/user/project/sub/directory"

	GitHub (Git)

		import "github.com/user/project"
		import "github.com/user/project/sub/directory"

	Google Code Project Hosting (Git, Mercurial, Subversion)

		import "project.googlecode.com/git"
		import "project.googlecode.com/git/sub/directory"

		import "project.googlecode.com/hg"
		import "project.googlecode.com/hg/sub/directory"

		import "project.googlecode.com/svn/trunk"
		import "project.googlecode.com/svn/trunk/sub/directory"

	Launchpad (Bazaar)

		import "launchpad.net/project"
		import "launchpad.net/project/series"
		import "launchpad.net/project/series/sub/directory"

		import "launchpad.net/~user/project/branch"
		import "launchpad.net/~user/project/branch/sub/directory"

For code hosted on other servers, an import path of the form

	repository.vcs/path

specifies the given repository, with or without the .vcs suffix,
using the named version control system, and then the path inside
that repository.  The supported version control systems are:

	Bazaar      .bzr
	Git         .git
	Mercurial   .hg
	Subversion  .svn

For example,

	import "example.org/user/foo.hg"

denotes the root directory of the Mercurial repository at
example.org/user/foo or foo.hg, and

	import "example.org/repo.git/foo/bar"

denotes the foo/bar directory of the Git repository at
example.com/repo or repo.git.

When a version control system supports multiple protocols,
each is tried in turn when downloading.  For example, a Git
download tries git://, then https://, then http://.

New downloaded packages are written to the first directory
listed in the GOPATH environment variable (see 'go help gopath').

The go command attempts to download the version of the
package appropriate for the Go release being used.
Run 'go help install' for more.
	`,
}

var helpGopath = &Command{
	UsageLine: "gopath",
	Short:     "GOPATH environment variable",
	Long: `
The GOPATH environment variable lists places to look for Go code.
On Unix, the value is a colon-separated string.
On Windows, the value is a semicolon-separated string.
On Plan 9, the value is a list.

GOPATH must be set to build and install packages outside the
standard Go tree.

Each directory listed in GOPATH must have a prescribed structure:

The src/ directory holds source code.  The path below 'src'
determines the import path or executable name.

The pkg/ directory holds installed package objects.
As in the Go tree, each target operating system and
architecture pair has its own subdirectory of pkg
(pkg/GOOS_GOARCH).

If DIR is a directory listed in the GOPATH, a package with
source in DIR/src/foo/bar can be imported as "foo/bar" and
has its compiled form installed to "DIR/pkg/GOOS_GOARCH/foo/bar.a".

The bin/ directory holds compiled commands.
Each command is named for its source directory, but only
the final element, not the entire path.  That is, the
command with source in DIR/src/foo/quux is installed into
DIR/bin/quux, not DIR/bin/foo/quux.  The foo/ is stripped
so that you can add DIR/bin to your PATH to get at the
installed commands.

Here's an example directory layout:

    GOPATH=/home/user/gocode

    /home/user/gocode/
        src/
            foo/
                bar/               (go code in package bar)
                    x.go
                quux/              (go code in package main)
                    y.go
        bin/
            quux                   (installed command)
        pkg/
            linux_amd64/
                foo/
                    bar.a          (installed package object)

Go searches each directory listed in GOPATH to find source code,
but new packages are always downloaded into the first directory 
in the list.
	`,
}
