// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package httpmux

import (
	"go/ast"
	"go/constant"
	"go/types"
	"regexp"
	"strings"

	"golang.org/x/mod/semver"
	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/analysis/passes/internal/analysisutil"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/go/types/typeutil"
)

const Doc = `report using Go 1.22 enhanced ServeMux patterns in older Go versions

The httpmux analysis is active for Go modules configured to run with Go 1.21 or
earlier versions. It reports calls to net/http.ServeMux.Handle and HandleFunc
methods whose patterns use features added in Go 1.22, like HTTP methods (such as
"GET") and wildcards. (See https://pkg.go.dev/net/http#ServeMux for details.)
Such patterns can be registered in older versions of Go, but will not behave as expected.`

var Analyzer = &analysis.Analyzer{
	Name:     "httpmux",
	Doc:      Doc,
	URL:      "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/httpmux",
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Run:      run,
}

var inTest bool // So Go version checks can be skipped during testing.

func run(pass *analysis.Pass) (any, error) {
	if !inTest {
		// Check that Go version is 1.21 or earlier.
		if goVersionAfter121(goVersion(pass.Pkg)) {
			return nil, nil
		}
	}
	if !analysisutil.Imports(pass.Pkg, "net/http") {
		return nil, nil
	}
	// Look for calls to ServeMux.Handle or ServeMux.HandleFunc.
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)

	nodeFilter := []ast.Node{
		(*ast.CallExpr)(nil),
	}

	inspect.Preorder(nodeFilter, func(n ast.Node) {
		call := n.(*ast.CallExpr)
		if isServeMuxRegisterCall(pass, call) {
			pat, ok := stringConstantExpr(pass, call.Args[0])
			if ok && likelyEnhancedPattern(pat) {
				pass.ReportRangef(call.Args[0], "possible enhanced ServeMux pattern used with Go version before 1.22 (update go.mod file?)")
			}
		}
	})
	return nil, nil
}

// isServeMuxRegisterCall reports whether call is a static call to one of:
// - net/http.Handle
// - net/http.HandleFunc
// - net/http.ServeMux.Handle
// - net/http.ServeMux.HandleFunc
// TODO(jba): consider expanding this to accommodate wrappers around these functions.
func isServeMuxRegisterCall(pass *analysis.Pass, call *ast.CallExpr) bool {
	fn := typeutil.StaticCallee(pass.TypesInfo, call)
	if fn == nil {
		return false
	}
	if analysisutil.IsFunctionNamed(fn, "net/http", "Handle", "HandleFunc") {
		return true
	}
	if !isMethodNamed(fn, "net/http", "Handle", "HandleFunc") {
		return false
	}
	t, ok := fn.Type().(*types.Signature).Recv().Type().(*types.Pointer)
	if !ok {
		return false
	}
	return analysisutil.IsNamedType(t.Elem(), "net/http", "ServeMux")
}

func isMethodNamed(f *types.Func, pkgPath string, names ...string) bool {
	if f == nil {
		return false
	}
	if f.Pkg() == nil || f.Pkg().Path() != pkgPath {
		return false
	}
	if f.Type().(*types.Signature).Recv() == nil {
		return false
	}
	for _, n := range names {
		if f.Name() == n {
			return true
		}
	}
	return false
}

// stringConstantExpr returns expression's string constant value.
//
// ("", false) is returned if expression isn't a string
// constant.
func stringConstantExpr(pass *analysis.Pass, expr ast.Expr) (string, bool) {
	lit := pass.TypesInfo.Types[expr].Value
	if lit != nil && lit.Kind() == constant.String {
		return constant.StringVal(lit), true
	}
	return "", false
}

// A valid wildcard must start a segment, and its name must be valid Go
// identifier.
var wildcardRegexp = regexp.MustCompile(`/\{[_\pL][_\pL\p{Nd}]*(\.\.\.)?\}`)

// likelyEnhancedPattern reports whether the ServeMux pattern pat probably
// contains either an HTTP method name or a wildcard, extensions added in Go 1.22.
func likelyEnhancedPattern(pat string) bool {
	if strings.Contains(pat, " ") {
		// A space in the pattern suggests that it begins with an HTTP method.
		return true
	}
	return wildcardRegexp.MatchString(pat)
}

func goVersionAfter121(goVersion string) bool {
	if goVersion == "" { // Maybe the stdlib?
		return true
	}
	version := versionFromGoVersion(goVersion)
	return semver.Compare(version, "v1.21") > 0
}

func goVersion(pkg *types.Package) string {
	// types.Package.GoVersion did not exist before Go 1.21.
	if p, ok := any(pkg).(interface{ GoVersion() string }); ok {
		return p.GoVersion()
	}
	return ""
}

var (
	// Regexp for matching go tags. The groups are:
	// 1  the major.minor version
	// 2  the patch version, or empty if none
	// 3  the entire prerelease, if present
	// 4  the prerelease type ("beta" or "rc")
	// 5  the prerelease number
	tagRegexp = regexp.MustCompile(`^go(\d+\.\d+)(\.\d+|)((beta|rc)(\d+))?$`)
)

// Copied from pkgsite/internal/stdlib.VersionForTag.
func versionFromGoVersion(goVersion string) string {
	// Special cases for go1.
	if goVersion == "go1" {
		return "v1.0.0"
	}
	if goVersion == "go1.0" {
		return ""
	}
	m := tagRegexp.FindStringSubmatch(goVersion)
	if m == nil {
		return ""
	}
	version := "v" + m[1]
	if m[2] != "" {
		version += m[2]
	} else {
		version += ".0"
	}
	if m[3] != "" {
		version += "-" + m[4] + "." + m[5]
	}
	return version
}
