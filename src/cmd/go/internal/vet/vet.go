// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package vet implements the ``go vet'' command.
package vet

import (
	"path/filepath"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/load"
	"cmd/go/internal/str"
)

var CmdVet = &base.Command{
	Run:         runVet,
	CustomFlags: true,
	UsageLine:   "vet [-n] [-x] [build flags] [vet flags] [packages]",
	Short:       "run go tool vet on packages",
	Long: `
Vet runs the Go vet command on the packages named by the import paths.

For more about vet and its flags, see 'go doc cmd/vet'.
For more about specifying packages, see 'go help packages'.

The -n flag prints commands that would be executed.
The -x flag prints commands as they are executed.

For more about build flags, see 'go help build'.

See also: go fmt, go fix.
	`,
}

func runVet(cmd *base.Command, args []string) {
	vetFlags, packages := vetFlags(args)
	for _, p := range load.Packages(packages) {
		// Vet expects to be given a set of files all from the same package.
		// Run once for package p and once for package p_test.
		if len(p.GoFiles)+len(p.CgoFiles)+len(p.TestGoFiles) > 0 {
			runVetFiles(p, vetFlags, str.StringList(p.GoFiles, p.CgoFiles, p.TestGoFiles, p.SFiles))
		}
		if len(p.XTestGoFiles) > 0 {
			runVetFiles(p, vetFlags, str.StringList(p.XTestGoFiles))
		}
	}
}

func runVetFiles(p *load.Package, flags, files []string) {
	for i := range files {
		files[i] = filepath.Join(p.Dir, files[i])
	}
	base.Run(cfg.BuildToolexec, base.Tool("vet"), flags, base.RelPaths(files))
}
