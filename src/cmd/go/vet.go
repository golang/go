// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "path/filepath"

func init() {
	addBuildFlags(cmdVet)
}

var cmdVet = &Command{
	Run:       runVet,
	UsageLine: "vet [-n] [-x] [build flags] [packages]",
	Short:     "run go tool vet on packages",
	Long: `
Vet runs the Go vet command on the packages named by the import paths.

For more about vet, see 'go doc cmd/vet'.
For more about specifying packages, see 'go help packages'.

To run the vet tool with specific options, run 'go tool vet'.

The -n flag prints commands that would be executed.
The -x flag prints commands as they are executed.

For more about build flags, see 'go help build'.

See also: go fmt, go fix.
	`,
}

func runVet(cmd *Command, args []string) {
	for _, p := range packages(args) {
		// Vet expects to be given a set of files all from the same package.
		// Run once for package p and once for package p_test.
		if len(p.GoFiles)+len(p.CgoFiles)+len(p.TestGoFiles) > 0 {
			runVetFiles(p, stringList(p.GoFiles, p.CgoFiles, p.TestGoFiles, p.SFiles))
		}
		if len(p.XTestGoFiles) > 0 {
			runVetFiles(p, stringList(p.XTestGoFiles))
		}
	}
}

func runVetFiles(p *Package, files []string) {
	for i := range files {
		files[i] = filepath.Join(p.Dir, files[i])
	}
	run(buildToolExec, tool("vet"), relPaths(files))
}
