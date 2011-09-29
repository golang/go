// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var cmdGet = &Command{
	Run:       runGet,
	UsageLine: "get [importpath...]",
	Short:     "download and install packages and dependencies",
	Long: `
Get downloads and installs the packages named by the import paths,
along with their dependencies.

After downloading the code, 'go get' looks for a tag beginning
with "go." that corresponds to the local Go version.
For Go "release.r58" it looks for a tag named "go.r58".
For "weekly.2011-06-03" it looks for "go.weekly.2011-06-03".
If the specific "go.X" tag is not found, it uses the latest earlier
version it can find.  Otherwise, it uses the default version for
the version control system: HEAD for git, tip for Mercurial,
and so on.

TODO: Explain versions better.

For more about import paths, see 'go help importpath'.

For more about how 'go get' finds source code to
download, see 'go help remote'.

See also: go build, go install, go clean.
	`,
}

func runGet(cmd *Command, args []string) {
	args = importPaths(args)
	_ = args
	panic("get not implemented")
}
