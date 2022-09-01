// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.18
// +build go1.18

package govulncheck

import (
	"fmt"
	"strings"

	"golang.org/x/mod/semver"
	isem "golang.org/x/tools/gopls/internal/govulncheck/semver"
	"golang.org/x/vuln/osv"
	"golang.org/x/vuln/vulncheck"
)

// LatestFixed returns the latest fixed version in the list of affected ranges,
// or the empty string if there are no fixed versions.
func LatestFixed(as []osv.Affected) string {
	v := ""
	for _, a := range as {
		for _, r := range a.Ranges {
			if r.Type == osv.TypeSemver {
				for _, e := range r.Events {
					if e.Fixed != "" && (v == "" ||
						semver.Compare(isem.CanonicalizeSemverPrefix(e.Fixed), isem.CanonicalizeSemverPrefix(v)) > 0) {
						v = e.Fixed
					}
				}
			}
		}
	}
	return v
}

// SummarizeCallStack returns a short description of the call stack.
// It uses one of two forms, depending on what the lowest function F in topPkgs
// calls:
//   - If it calls a function V from the vulnerable package, then summarizeCallStack
//     returns "F calls V".
//   - If it calls a function G in some other package, which eventually calls V,
//     it returns "F calls G, which eventually calls V".
//
// If it can't find any of these functions, summarizeCallStack returns the empty string.
func SummarizeCallStack(cs vulncheck.CallStack, topPkgs map[string]bool, vulnPkg string) string {
	// Find the lowest function in the top packages.
	iTop := lowest(cs, func(e vulncheck.StackEntry) bool {
		return topPkgs[PkgPath(e.Function)]
	})
	if iTop < 0 {
		return ""
	}
	// Find the highest function in the vulnerable package that is below iTop.
	iVuln := highest(cs[iTop+1:], func(e vulncheck.StackEntry) bool {
		return PkgPath(e.Function) == vulnPkg
	})
	if iVuln < 0 {
		return ""
	}
	iVuln += iTop + 1 // adjust for slice in call to highest.
	topName := FuncName(cs[iTop].Function)
	topPos := AbsRelShorter(FuncPos(cs[iTop].Call))
	if topPos != "" {
		topPos += ": "
	}
	vulnName := FuncName(cs[iVuln].Function)
	if iVuln == iTop+1 {
		return fmt.Sprintf("%s%s calls %s", topPos, topName, vulnName)
	}
	return fmt.Sprintf("%s%s calls %s, which eventually calls %s",
		topPos, topName, FuncName(cs[iTop+1].Function), vulnName)
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

// PkgPath returns the package path from fn.
func PkgPath(fn *vulncheck.FuncNode) string {
	if fn.PkgPath != "" {
		return fn.PkgPath
	}
	s := strings.TrimPrefix(fn.RecvType, "*")
	if i := strings.LastIndexByte(s, '.'); i > 0 {
		s = s[:i]
	}
	return s
}

// FuncName returns the function name from fn, adjusted
// to remove pointer annotations.
func FuncName(fn *vulncheck.FuncNode) string {
	return strings.TrimPrefix(fn.String(), "*")
}

// FuncPos returns the function position from call.
func FuncPos(call *vulncheck.CallSite) string {
	if call != nil && call.Pos != nil {
		return call.Pos.String()
	}
	return ""
}
