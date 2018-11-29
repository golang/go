// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modload

import (
	"fmt"
	"os"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/modinfo"
	"cmd/go/internal/module"
	"cmd/go/internal/par"
	"cmd/go/internal/search"
)

func ListModules(args []string, listU, listVersions bool) []*modinfo.ModulePublic {
	mods := listModules(args)
	if listU || listVersions {
		var work par.Work
		for _, m := range mods {
			work.Add(m)
			if m.Replace != nil {
				work.Add(m.Replace)
			}
		}
		work.Do(10, func(item interface{}) {
			m := item.(*modinfo.ModulePublic)
			if listU {
				addUpdate(m)
			}
			if listVersions {
				addVersions(m)
			}
		})
	}
	return mods
}

func listModules(args []string) []*modinfo.ModulePublic {
	LoadBuildList()
	if len(args) == 0 {
		return []*modinfo.ModulePublic{moduleInfo(buildList[0], true)}
	}

	var mods []*modinfo.ModulePublic
	matchedBuildList := make([]bool, len(buildList))
	for _, arg := range args {
		if strings.Contains(arg, `\`) {
			base.Fatalf("go: module paths never use backslash")
		}
		if search.IsRelativePath(arg) {
			base.Fatalf("go: cannot use relative path %s to specify module", arg)
		}
		if i := strings.Index(arg, "@"); i >= 0 {
			info, err := Query(arg[:i], arg[i+1:], nil)
			if err != nil {
				mods = append(mods, &modinfo.ModulePublic{
					Path:    arg[:i],
					Version: arg[i+1:],
					Error: &modinfo.ModuleError{
						Err: err.Error(),
					},
				})
				continue
			}
			mods = append(mods, moduleInfo(module.Version{Path: arg[:i], Version: info.Version}, false))
			continue
		}

		// Module path or pattern.
		var match func(string) bool
		var literal bool
		if arg == "all" {
			match = func(string) bool { return true }
		} else if strings.Contains(arg, "...") {
			match = search.MatchPattern(arg)
		} else {
			match = func(p string) bool { return arg == p }
			literal = true
		}
		matched := false
		for i, m := range buildList {
			if match(m.Path) {
				matched = true
				if !matchedBuildList[i] {
					matchedBuildList[i] = true
					mods = append(mods, moduleInfo(m, true))
				}
			}
		}
		if !matched {
			if literal {
				mods = append(mods, &modinfo.ModulePublic{
					Path: arg,
					Error: &modinfo.ModuleError{
						Err: fmt.Sprintf("module %q is not a known dependency", arg),
					},
				})
			} else {
				fmt.Fprintf(os.Stderr, "warning: pattern %q matched no module dependencies\n", arg)
			}
		}
	}

	return mods
}
