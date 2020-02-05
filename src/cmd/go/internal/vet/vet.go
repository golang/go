// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package vet implements the ``go vet'' command.
package vet

import (
	"cmd/go/internal/base"
	"cmd/go/internal/load"
	"cmd/go/internal/modload"
	"cmd/go/internal/work"
	"path/filepath"
)

var CmdVet = &base.Command{
	Run:         runVet,
	CustomFlags: true,
	UsageLine:   "go vet [-n] [-x] [-vettool prog] [build flags] [vet flags] [packages]",
	Short:       "report likely mistakes in packages",
	Long: `
Vet runs the Go vet command on the packages named by the import paths.

For more about vet and its flags, see 'go doc cmd/vet'.
For more about specifying packages, see 'go help packages'.
For a list of checkers and their flags, see 'go tool vet help'.
For details of a specific checker such as 'printf', see 'go tool vet help printf'.

The -n flag prints commands that would be executed.
The -x flag prints commands as they are executed.

The -vettool=prog flag selects a different analysis tool with alternative
or additional checks.
For example, the 'shadow' analyzer can be built and run using these commands:

  go install golang.org/x/tools/go/analysis/passes/shadow/cmd/shadow
  go vet -vettool=$(which shadow)

The build flags supported by go vet are those that control package resolution
and execution, such as -n, -x, -v, -tags, and -toolexec.
For more about these flags, see 'go help build'.

See also: go fmt, go fix.
	`,
}

func runVet(cmd *base.Command, args []string) {
	modload.LoadTests = true

	vetFlags, pkgArgs := vetFlags(vetUsage, args)

	work.BuildInit()
	work.VetFlags = vetFlags
	if len(vetFlags) > 0 {
		work.VetExplicit = true
	}
	if vetTool != "" {
		var err error
		work.VetTool, err = filepath.Abs(vetTool)
		if err != nil {
			base.Fatalf("%v", err)
		}
	}

	pkgs := load.PackagesForBuild(pkgArgs)
	if len(pkgs) == 0 {
		base.Fatalf("no packages to vet")
	}

	var b work.Builder
	b.Init()

	root := &work.Action{Mode: "go vet"}
	for _, p := range pkgs {
		_, ptest, pxtest, err := load.TestPackagesFor(p, nil)
		if err != nil {
			base.Errorf("%v", err)
			continue
		}
		if len(ptest.GoFiles) == 0 && len(ptest.CgoFiles) == 0 && pxtest == nil {
			base.Errorf("go vet %s: no Go files in %s", p.ImportPath, p.Dir)
			continue
		}
		if len(ptest.GoFiles) > 0 || len(ptest.CgoFiles) > 0 {
			root.Deps = append(root.Deps, b.VetAction(work.ModeBuild, work.ModeBuild, ptest))
		}
		if pxtest != nil {
			root.Deps = append(root.Deps, b.VetAction(work.ModeBuild, work.ModeBuild, pxtest))
		}
	}
	b.Do(root)
}
