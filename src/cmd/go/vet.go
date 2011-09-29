// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var cmdVet = &Command{
	Run:       runVet,
	UsageLine: "vet [importpath...]",
	Short:     "run govet on packages",
	Long: `
Vet runs the govet command on the packages named by the import paths.

For more about govet, see 'godoc govet'.
For more about import paths, see 'go help importpath'.

To run govet with specific options, run govet itself.

See also: go fmt, go fix.
	`,
}

func runVet(cmd *Command, args []string) {
	args = importPaths(args)
	_ = args
	panic("vet not implemented")
}
