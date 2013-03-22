// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	addBuildFlagsNX(cmdVet)
}

var cmdVet = &Command{
	Run:       runVet,
	UsageLine: "vet [-n] [-x] [packages]",
	Short:     "run go tool vet on packages",
	Long: `
Vet runs the Go vet command on the packages named by the import paths.

For more about vet, see 'godoc vet'.
For more about specifying packages, see 'go help packages'.

To run the vet tool with specific options, run 'go tool vet'.

The -n flag prints commands that would be executed.
The -x flag prints commands as they are executed.

See also: go fmt, go fix.
	`,
}

func runVet(cmd *Command, args []string) {
	for _, pkg := range packages(args) {
		// Use pkg.gofiles instead of pkg.Dir so that
		// the command only applies to this package,
		// not to packages in subdirectories.
		run(tool("vet"), relPaths(stringList(pkg.gofiles, pkg.sfiles)))
	}
}
