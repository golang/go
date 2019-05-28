// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go mod tidy

package modcmd

import (
	"fmt"
	"os"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/modfetch"
	"cmd/go/internal/modload"
	"cmd/go/internal/module"
)

var cmdTidy = &base.Command{
	UsageLine: "go mod tidy [-v]",
	Short:     "add missing and remove unused modules",
	Long: `
Tidy makes sure go.mod matches the source code in the module.
It adds any missing modules necessary to build the current module's
packages and dependencies, and it removes unused modules that
don't provide any relevant packages. It also adds any missing entries
to go.sum and removes any unnecessary ones.

The -v flag causes tidy to print information about removed modules
to standard error.
	`,
}

func init() {
	cmdTidy.Run = runTidy // break init cycle
	cmdTidy.Flag.BoolVar(&cfg.BuildV, "v", false, "")
}

func runTidy(cmd *base.Command, args []string) {
	if len(args) > 0 {
		base.Fatalf("go mod tidy: no arguments allowed")
	}

	// LoadALL adds missing modules.
	// Remove unused modules.
	used := make(map[module.Version]bool)
	for _, pkg := range modload.LoadALL() {
		used[modload.PackageModule(pkg)] = true
	}
	used[modload.Target] = true // note: LoadALL initializes Target

	inGoMod := make(map[string]bool)
	for _, r := range modload.ModFile().Require {
		inGoMod[r.Mod.Path] = true
	}

	var keep []module.Version
	for _, m := range modload.BuildList() {
		if used[m] {
			keep = append(keep, m)
		} else if cfg.BuildV && inGoMod[m.Path] {
			fmt.Fprintf(os.Stderr, "unused %s\n", m.Path)
		}
	}
	modload.SetBuildList(keep)
	modTidyGoSum() // updates memory copy; WriteGoMod on next line flushes it out
	modload.WriteGoMod()
}

// modTidyGoSum resets the go.sum file content
// to be exactly what's needed for the current go.mod.
func modTidyGoSum() {
	// Assuming go.sum already has at least enough from the successful load,
	// we only have to tell modfetch what needs keeping.
	reqs := modload.Reqs()
	keep := make(map[module.Version]bool)
	replaced := make(map[module.Version]bool)
	var walk func(module.Version)
	walk = func(m module.Version) {
		// If we build using a replacement module, keep the sum for the replacement,
		// since that's the code we'll actually use during a build.
		//
		// TODO(golang.org/issue/29182): Perhaps we should keep both sums, and the
		// sums for both sets of transitive requirements.
		r := modload.Replacement(m)
		if r.Path == "" {
			keep[m] = true
		} else {
			keep[r] = true
			replaced[m] = true
		}
		list, _ := reqs.Required(m)
		for _, r := range list {
			if !keep[r] && !replaced[r] {
				walk(r)
			}
		}
	}
	walk(modload.Target)
	modfetch.TrimGoSum(keep)
}
