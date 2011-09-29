// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var cmdClean = &Command{
	Run:       runClean,
	UsageLine: "clean [-nuke] [importpath...]",
	Short:     "remove intermediate objects",
	Long: `
Clean removes intermediate object files generated during
the compilation of the packages named by the import paths,
but by default it does not remove the installed package binaries.

The -nuke flag causes clean to remove the installed package binaries too.

TODO: Clean does not clean dependencies of the packages.
TODO: Rename -nuke.

For more about import paths, see 'go help importpath'.
	`,
}

var cleanNuke = cmdClean.Flag.Bool("nuke", false, "")

func runClean(cmd *Command, args []string) {
	args = importPaths(args)
	_ = args
	panic("nuke not implemented")
}
