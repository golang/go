// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package fix implements the “go fix” command.
package fix

import (
	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/load"
	"cmd/go/internal/modload"
	"cmd/go/internal/str"
	"cmd/go/internal/work"
	"context"
	"fmt"
	"go/build"
	"os"
)

var CmdFix = &base.Command{
	UsageLine: "go fix [-fix list] [packages]",
	Short:     "update packages to use new APIs",
	Long: `
Fix runs the Go fix command on the packages named by the import paths.

The -fix flag sets a comma-separated list of fixes to run.
The default is all known fixes.
(Its value is passed to 'go tool fix -r'.)

For more about fix, see 'go doc cmd/fix'.
For more about specifying packages, see 'go help packages'.

To run fix with other options, run 'go tool fix'.

See also: go fmt, go vet.
	`,
}

var fixes = CmdFix.Flag.String("fix", "", "comma-separated list of fixes to apply")

func init() {
	work.AddBuildFlags(CmdFix, work.DefaultBuildFlags)
	CmdFix.Run = runFix // fix cycle
}

func runFix(ctx context.Context, cmd *base.Command, args []string) {
	pkgs := load.PackagesAndErrors(ctx, load.PackageOpts{}, args)
	w := 0
	for _, pkg := range pkgs {
		if pkg.Error != nil {
			base.Errorf("%v", pkg.Error)
			continue
		}
		pkgs[w] = pkg
		w++
	}
	pkgs = pkgs[:w]

	printed := false
	for _, pkg := range pkgs {
		if modload.Enabled() && pkg.Module != nil && !pkg.Module.Main {
			if !printed {
				fmt.Fprintf(os.Stderr, "go: not fixing packages in dependency modules\n")
				printed = true
			}
			continue
		}
		// Use pkg.gofiles instead of pkg.Dir so that
		// the command only applies to this package,
		// not to packages in subdirectories.
		files := base.RelPaths(pkg.InternalAllGoFiles())
		goVersion := ""
		if pkg.Module != nil {
			goVersion = "go" + pkg.Module.GoVersion
		} else if pkg.Standard {
			goVersion = build.Default.ReleaseTags[len(build.Default.ReleaseTags)-1]
		}
		var fixArg []string
		if *fixes != "" {
			fixArg = []string{"-r=" + *fixes}
		}
		base.Run(str.StringList(cfg.BuildToolexec, base.Tool("fix"), "-go="+goVersion, fixArg, files))
	}
}
