// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var cmdFix = &Command{
	Run:       runFix,
	UsageLine: "fix [importpath...]",
	Short:     "run gofix on packages",
	Long: `
Fix runs the gofix command on the packages named by the import paths.

For more about gofix, see 'godoc gofix'.
For more about import paths, see 'go help importpath'.

To run gofix with specific options, run gofix itself.

See also: go fmt, go vet.
	`,
}

func runFix(cmd *Command, args []string) {
	for _, pkg := range packages(args) {
		// Use pkg.gofiles instead of pkg.Dir so that
		// the command only applies to this package,
		// not to packages in subdirectories.
		run(append([]string{"gofix"}, pkg.gofiles...)...)
	}
}
