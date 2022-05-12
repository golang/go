// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.18
// +build go1.18

package vulncheck

import (
	"fmt"
	"go/token"
	"sort"
	"strings"

	"golang.org/x/mod/semver"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/vuln/osv"
	"golang.org/x/vuln/vulncheck"
)

// TODO(hyangah): automate copy of golang.org/x/vuln/cmd.

// moduleVersionMap builds a map from module paths to versions.
func moduleVersionMap(mods []*vulncheck.Module) map[string]string {
	moduleVersions := map[string]string{}
	for _, m := range mods {
		v := m.Version
		if m.Replace != nil {
			v = m.Replace.Version
		}
		moduleVersions[m.Path] = v
	}
	return moduleVersions
}

func groupByIDAndPackage(vs []*vulncheck.Vuln) [][]*vulncheck.Vuln {
	groups := map[[2]string][]*vulncheck.Vuln{}
	for _, v := range vs {
		key := [2]string{v.OSV.ID, v.PkgPath}
		groups[key] = append(groups[key], v)
	}

	var res [][]*vulncheck.Vuln
	for _, g := range groups {
		res = append(res, g)
	}
	sort.Slice(res, func(i, j int) bool {
		return res[i][0].PkgPath < res[j][0].PkgPath
	})
	return res
}

// latestFixed returns the latest fixed version in the list of affected ranges,
// or the empty string if there are no fixed versions.
func latestFixed(as []osv.Affected) string {
	v := ""
	for _, a := range as {
		for _, r := range a.Ranges {
			if r.Type == osv.TypeSemver {
				for _, e := range r.Events {
					if e.Fixed != "" && (v == "" || semver.Compare(e.Fixed, v) > 0) {
						v = e.Fixed
					}
				}
			}
		}
	}
	if v == "" || v[0] == 'v' {
		return v
	}
	return "v" + v
}

// summarizeCallStack returns a short description of the call stack.
// It uses one of two forms, depending on what the lowest function F in topPkgs
// calls:
//   - If it calls a function V from the vulnerable package, then summarizeCallStack
//     returns "F calls V".
//   - If it calls a function G in some other package, which eventually calls V,
//     it returns "F calls G, which eventually calls V".
//
// If it can't find any of these functions, summarizeCallStack returns the empty string.
func summarizeCallStack(cs vulncheck.CallStack, topPkgs map[string]bool, vulnPkg string) string {
	// Find the lowest function in the top packages.
	iTop := lowest(cs, func(e vulncheck.StackEntry) bool {
		return topPkgs[pkgPath(e.Function)]
	})
	if iTop < 0 {
		return ""
	}
	// Find the highest function in the vulnerable package that is below iTop.
	iVuln := highest(cs[iTop+1:], func(e vulncheck.StackEntry) bool {
		return pkgPath(e.Function) == vulnPkg
	})
	if iVuln < 0 {
		return ""
	}
	iVuln += iTop + 1 // adjust for slice in call to highest.
	topName := funcName(cs[iTop].Function)
	vulnName := funcName(cs[iVuln].Function)
	if iVuln == iTop+1 {
		return fmt.Sprintf("%s calls %s", topName, vulnName)
	}
	return fmt.Sprintf("%s calls %s, which eventually calls %s",
		topName, funcName(cs[iTop+1].Function), vulnName)
}

// highest returns the highest (one with the smallest index) entry in the call
// stack for which f returns true.
func highest(cs vulncheck.CallStack, f func(e vulncheck.StackEntry) bool) int {
	for i := 0; i < len(cs); i++ {
		if f(cs[i]) {
			return i
		}
	}
	return -1
}

// lowest returns the lowest (one with the largets index) entry in the call
// stack for which f returns true.
func lowest(cs vulncheck.CallStack, f func(e vulncheck.StackEntry) bool) int {
	for i := len(cs) - 1; i >= 0; i-- {
		if f(cs[i]) {
			return i
		}
	}
	return -1
}
func pkgPath(fn *vulncheck.FuncNode) string {
	if fn.PkgPath != "" {
		return fn.PkgPath
	}
	s := strings.TrimPrefix(fn.RecvType, "*")
	if i := strings.LastIndexByte(s, '.'); i > 0 {
		s = s[:i]
	}
	return s
}

func toCallStack(src vulncheck.CallStack) CallStack {
	var dest []StackEntry
	for _, e := range src {
		dest = append(dest, toStackEntry(e))
	}
	return dest
}

func toStackEntry(src vulncheck.StackEntry) StackEntry {
	f, call := src.Function, src.Call
	pos := f.Pos
	desc := funcName(f)
	if src.Call != nil {
		pos = src.Call.Pos // Exact call site position is helpful.
		if !call.Resolved {
			// In case of a statically unresolved call site, communicate to the client
			// that this was approximately resolved to f

			desc += " [approx.]"
		}
	}
	return StackEntry{
		Name: desc,
		URI:  filenameToURI(pos),
		Pos:  posToPosition(pos),
	}
}

func funcName(fn *vulncheck.FuncNode) string {
	return strings.TrimPrefix(fn.String(), "*")
}

// href returns a URL embedded in the entry if any.
// If no suitable URL is found, it returns a default entry in
// pkg.go.dev/vuln.
func href(vuln *osv.Entry) string {
	for _, affected := range vuln.Affected {
		if url := affected.DatabaseSpecific.URL; url != "" {
			return url
		}
	}
	for _, r := range vuln.References {
		if r.Type == "WEB" {
			return r.URL
		}
	}
	return fmt.Sprintf("https://pkg.go.dev/vuln/%s", vuln.ID)
}

func filenameToURI(pos *token.Position) protocol.DocumentURI {
	if pos == nil || pos.Filename == "" {
		return ""
	}
	return protocol.URIFromPath(pos.Filename)
}

func posToPosition(pos *token.Position) (p protocol.Position) {
	// token.Position.Line starts from 1, and
	// LSP protocol's position line is 0-based.
	if pos != nil {
		p.Line = uint32(pos.Line - 1)
		// TODO(hyangah): LSP uses UTF16 column.
		// We need utility like span.ToUTF16Column,
		// but somthing that does not require file contents.
	}
	return p
}
