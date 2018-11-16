// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package search

import (
	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"fmt"
	"go/build"
	"log"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"
)

// A Match represents the result of matching a single package pattern.
type Match struct {
	Pattern string   // the pattern itself
	Literal bool     // whether it is a literal (no wildcards)
	Pkgs    []string // matching packages (dirs or import paths)
}

// MatchPackages returns all the packages that can be found
// under the $GOPATH directories and $GOROOT matching pattern.
// The pattern is either "all" (all packages), "std" (standard packages),
// "cmd" (standard commands), or a path including "...".
func MatchPackages(pattern string) *Match {
	m := &Match{
		Pattern: pattern,
		Literal: false,
	}
	match := func(string) bool { return true }
	treeCanMatch := func(string) bool { return true }
	if !IsMetaPackage(pattern) {
		match = MatchPattern(pattern)
		treeCanMatch = TreeCanMatchPattern(pattern)
	}

	have := map[string]bool{
		"builtin": true, // ignore pseudo-package that exists only for documentation
	}
	if !cfg.BuildContext.CgoEnabled {
		have["runtime/cgo"] = true // ignore during walk
	}

	for _, src := range cfg.BuildContext.SrcDirs() {
		if (pattern == "std" || pattern == "cmd") && src != cfg.GOROOTsrc {
			continue
		}
		src = filepath.Clean(src) + string(filepath.Separator)
		root := src
		if pattern == "cmd" {
			root += "cmd" + string(filepath.Separator)
		}
		filepath.Walk(root, func(path string, fi os.FileInfo, err error) error {
			if err != nil || path == src {
				return nil
			}

			want := true
			// Avoid .foo, _foo, and testdata directory trees.
			_, elem := filepath.Split(path)
			if strings.HasPrefix(elem, ".") || strings.HasPrefix(elem, "_") || elem == "testdata" {
				want = false
			}

			name := filepath.ToSlash(path[len(src):])
			if pattern == "std" && (!IsStandardImportPath(name) || name == "cmd") {
				// The name "std" is only the standard library.
				// If the name is cmd, it's the root of the command tree.
				want = false
			}
			if !treeCanMatch(name) {
				want = false
			}

			if !fi.IsDir() {
				if fi.Mode()&os.ModeSymlink != 0 && want {
					if target, err := os.Stat(path); err == nil && target.IsDir() {
						fmt.Fprintf(os.Stderr, "warning: ignoring symlink %s\n", path)
					}
				}
				return nil
			}
			if !want {
				return filepath.SkipDir
			}

			if have[name] {
				return nil
			}
			have[name] = true
			if !match(name) {
				return nil
			}
			pkg, err := cfg.BuildContext.ImportDir(path, 0)
			if err != nil {
				if _, noGo := err.(*build.NoGoError); noGo {
					return nil
				}
			}

			// If we are expanding "cmd", skip main
			// packages under cmd/vendor. At least as of
			// March, 2017, there is one there for the
			// vendored pprof tool.
			if pattern == "cmd" && strings.HasPrefix(pkg.ImportPath, "cmd/vendor") && pkg.Name == "main" {
				return nil
			}

			m.Pkgs = append(m.Pkgs, name)
			return nil
		})
	}
	return m
}

var modRoot string

func SetModRoot(dir string) {
	modRoot = dir
}

// MatchPackagesInFS is like allPackages but is passed a pattern
// beginning ./ or ../, meaning it should scan the tree rooted
// at the given directory. There are ... in the pattern too.
// (See go help packages for pattern syntax.)
func MatchPackagesInFS(pattern string) *Match {
	m := &Match{
		Pattern: pattern,
		Literal: false,
	}

	// Find directory to begin the scan.
	// Could be smarter but this one optimization
	// is enough for now, since ... is usually at the
	// end of a path.
	i := strings.Index(pattern, "...")
	dir, _ := path.Split(pattern[:i])

	// pattern begins with ./ or ../.
	// path.Clean will discard the ./ but not the ../.
	// We need to preserve the ./ for pattern matching
	// and in the returned import paths.
	prefix := ""
	if strings.HasPrefix(pattern, "./") {
		prefix = "./"
	}
	match := MatchPattern(pattern)

	if modRoot != "" {
		abs, err := filepath.Abs(dir)
		if err != nil {
			base.Fatalf("go: %v", err)
		}
		if !hasFilepathPrefix(abs, modRoot) {
			base.Fatalf("go: pattern %s refers to dir %s, outside module root %s", pattern, abs, modRoot)
			return nil
		}
	}

	filepath.Walk(dir, func(path string, fi os.FileInfo, err error) error {
		if err != nil || !fi.IsDir() {
			return nil
		}
		top := false
		if path == dir {
			// filepath.Walk starts at dir and recurses. For the recursive case,
			// the path is the result of filepath.Join, which calls filepath.Clean.
			// The initial case is not Cleaned, though, so we do this explicitly.
			//
			// This converts a path like "./io/" to "io". Without this step, running
			// "cd $GOROOT/src; go list ./io/..." would incorrectly skip the io
			// package, because prepending the prefix "./" to the unclean path would
			// result in "././io", and match("././io") returns false.
			top = true
			path = filepath.Clean(path)
		}

		// Avoid .foo, _foo, and testdata directory trees, but do not avoid "." or "..".
		_, elem := filepath.Split(path)
		dot := strings.HasPrefix(elem, ".") && elem != "." && elem != ".."
		if dot || strings.HasPrefix(elem, "_") || elem == "testdata" {
			return filepath.SkipDir
		}

		if !top && cfg.ModulesEnabled {
			// Ignore other modules found in subdirectories.
			if _, err := os.Stat(filepath.Join(path, "go.mod")); err == nil {
				return filepath.SkipDir
			}
		}

		name := prefix + filepath.ToSlash(path)
		if !match(name) {
			return nil
		}

		// We keep the directory if we can import it, or if we can't import it
		// due to invalid Go source files. This means that directories containing
		// parse errors will be built (and fail) instead of being silently skipped
		// as not matching the pattern. Go 1.5 and earlier skipped, but that
		// behavior means people miss serious mistakes.
		// See golang.org/issue/11407.
		if p, err := cfg.BuildContext.ImportDir(path, 0); err != nil && (p == nil || len(p.InvalidGoFiles) == 0) {
			if _, noGo := err.(*build.NoGoError); !noGo {
				log.Print(err)
			}
			return nil
		}
		m.Pkgs = append(m.Pkgs, name)
		return nil
	})
	return m
}

// TreeCanMatchPattern(pattern)(name) reports whether
// name or children of name can possibly match pattern.
// Pattern is the same limited glob accepted by matchPattern.
func TreeCanMatchPattern(pattern string) func(name string) bool {
	wildCard := false
	if i := strings.Index(pattern, "..."); i >= 0 {
		wildCard = true
		pattern = pattern[:i]
	}
	return func(name string) bool {
		return len(name) <= len(pattern) && hasPathPrefix(pattern, name) ||
			wildCard && strings.HasPrefix(name, pattern)
	}
}

// MatchPattern(pattern)(name) reports whether
// name matches pattern. Pattern is a limited glob
// pattern in which '...' means 'any string' and there
// is no other special syntax.
// Unfortunately, there are two special cases. Quoting "go help packages":
//
// First, /... at the end of the pattern can match an empty string,
// so that net/... matches both net and packages in its subdirectories, like net/http.
// Second, any slash-separted pattern element containing a wildcard never
// participates in a match of the "vendor" element in the path of a vendored
// package, so that ./... does not match packages in subdirectories of
// ./vendor or ./mycode/vendor, but ./vendor/... and ./mycode/vendor/... do.
// Note, however, that a directory named vendor that itself contains code
// is not a vendored package: cmd/vendor would be a command named vendor,
// and the pattern cmd/... matches it.
func MatchPattern(pattern string) func(name string) bool {
	// Convert pattern to regular expression.
	// The strategy for the trailing /... is to nest it in an explicit ? expression.
	// The strategy for the vendor exclusion is to change the unmatchable
	// vendor strings to a disallowed code point (vendorChar) and to use
	// "(anything but that codepoint)*" as the implementation of the ... wildcard.
	// This is a bit complicated but the obvious alternative,
	// namely a hand-written search like in most shell glob matchers,
	// is too easy to make accidentally exponential.
	// Using package regexp guarantees linear-time matching.

	const vendorChar = "\x00"

	if strings.Contains(pattern, vendorChar) {
		return func(name string) bool { return false }
	}

	re := regexp.QuoteMeta(pattern)
	re = replaceVendor(re, vendorChar)
	switch {
	case strings.HasSuffix(re, `/`+vendorChar+`/\.\.\.`):
		re = strings.TrimSuffix(re, `/`+vendorChar+`/\.\.\.`) + `(/vendor|/` + vendorChar + `/\.\.\.)`
	case re == vendorChar+`/\.\.\.`:
		re = `(/vendor|/` + vendorChar + `/\.\.\.)`
	case strings.HasSuffix(re, `/\.\.\.`):
		re = strings.TrimSuffix(re, `/\.\.\.`) + `(/\.\.\.)?`
	}
	re = strings.ReplaceAll(re, `\.\.\.`, `[^`+vendorChar+`]*`)

	reg := regexp.MustCompile(`^` + re + `$`)

	return func(name string) bool {
		if strings.Contains(name, vendorChar) {
			return false
		}
		return reg.MatchString(replaceVendor(name, vendorChar))
	}
}

// replaceVendor returns the result of replacing
// non-trailing vendor path elements in x with repl.
func replaceVendor(x, repl string) string {
	if !strings.Contains(x, "vendor") {
		return x
	}
	elem := strings.Split(x, "/")
	for i := 0; i < len(elem)-1; i++ {
		if elem[i] == "vendor" {
			elem[i] = repl
		}
	}
	return strings.Join(elem, "/")
}

// WarnUnmatched warns about patterns that didn't match any packages.
func WarnUnmatched(matches []*Match) {
	for _, m := range matches {
		if len(m.Pkgs) == 0 {
			fmt.Fprintf(os.Stderr, "go: warning: %q matched no packages\n", m.Pattern)
		}
	}
}

// ImportPaths returns the matching paths to use for the given command line.
// It calls ImportPathsQuiet and then WarnUnmatched.
func ImportPaths(patterns []string) []*Match {
	matches := ImportPathsQuiet(patterns)
	WarnUnmatched(matches)
	return matches
}

// ImportPathsQuiet is like ImportPaths but does not warn about patterns with no matches.
func ImportPathsQuiet(patterns []string) []*Match {
	var out []*Match
	for _, a := range CleanPatterns(patterns) {
		if IsMetaPackage(a) {
			out = append(out, MatchPackages(a))
			continue
		}
		if strings.Contains(a, "...") {
			if build.IsLocalImport(a) {
				out = append(out, MatchPackagesInFS(a))
			} else {
				out = append(out, MatchPackages(a))
			}
			continue
		}
		out = append(out, &Match{Pattern: a, Literal: true, Pkgs: []string{a}})
	}
	return out
}

// CleanPatterns returns the patterns to use for the given
// command line. It canonicalizes the patterns but does not
// evaluate any matches.
func CleanPatterns(patterns []string) []string {
	if len(patterns) == 0 {
		return []string{"."}
	}
	var out []string
	for _, a := range patterns {
		// Arguments are supposed to be import paths, but
		// as a courtesy to Windows developers, rewrite \ to /
		// in command-line arguments. Handles .\... and so on.
		if filepath.Separator == '\\' {
			a = strings.ReplaceAll(a, `\`, `/`)
		}

		// Put argument in canonical form, but preserve leading ./.
		if strings.HasPrefix(a, "./") {
			a = "./" + path.Clean(a)
			if a == "./." {
				a = "."
			}
		} else {
			a = path.Clean(a)
		}
		out = append(out, a)
	}
	return out
}

// IsMetaPackage checks if name is a reserved package name that expands to multiple packages.
func IsMetaPackage(name string) bool {
	return name == "std" || name == "cmd" || name == "all"
}

// hasPathPrefix reports whether the path s begins with the
// elements in prefix.
func hasPathPrefix(s, prefix string) bool {
	switch {
	default:
		return false
	case len(s) == len(prefix):
		return s == prefix
	case len(s) > len(prefix):
		if prefix != "" && prefix[len(prefix)-1] == '/' {
			return strings.HasPrefix(s, prefix)
		}
		return s[len(prefix)] == '/' && s[:len(prefix)] == prefix
	}
}

// hasFilepathPrefix reports whether the path s begins with the
// elements in prefix.
func hasFilepathPrefix(s, prefix string) bool {
	switch {
	default:
		return false
	case len(s) == len(prefix):
		return s == prefix
	case len(s) > len(prefix):
		if prefix != "" && prefix[len(prefix)-1] == filepath.Separator {
			return strings.HasPrefix(s, prefix)
		}
		return s[len(prefix)] == filepath.Separator && s[:len(prefix)] == prefix
	}
}

// IsStandardImportPath reports whether $GOROOT/src/path should be considered
// part of the standard distribution. For historical reasons we allow people to add
// their own code to $GOROOT instead of using $GOPATH, but we assume that
// code will start with a domain name (dot in the first element).
//
// Note that this function is meant to evaluate whether a directory found in GOROOT
// should be treated as part of the standard library. It should not be used to decide
// that a directory found in GOPATH should be rejected: directories in GOPATH
// need not have dots in the first element, and they just take their chances
// with future collisions in the standard library.
func IsStandardImportPath(path string) bool {
	i := strings.Index(path, "/")
	if i < 0 {
		i = len(path)
	}
	elem := path[:i]
	return !strings.Contains(elem, ".")
}

// IsRelativePath reports whether pattern should be interpreted as a directory
// path relative to the current directory, as opposed to a pattern matching
// import paths.
func IsRelativePath(pattern string) bool {
	return strings.HasPrefix(pattern, "./") || strings.HasPrefix(pattern, "../") || pattern == "." || pattern == ".."
}

// InDir checks whether path is in the file tree rooted at dir.
// If so, InDir returns an equivalent path relative to dir.
// If not, InDir returns an empty string.
// InDir makes some effort to succeed even in the presence of symbolic links.
// TODO(rsc): Replace internal/test.inDir with a call to this function for Go 1.12.
func InDir(path, dir string) string {
	if rel := inDirLex(path, dir); rel != "" {
		return rel
	}
	xpath, err := filepath.EvalSymlinks(path)
	if err != nil || xpath == path {
		xpath = ""
	} else {
		if rel := inDirLex(xpath, dir); rel != "" {
			return rel
		}
	}

	xdir, err := filepath.EvalSymlinks(dir)
	if err == nil && xdir != dir {
		if rel := inDirLex(path, xdir); rel != "" {
			return rel
		}
		if xpath != "" {
			if rel := inDirLex(xpath, xdir); rel != "" {
				return rel
			}
		}
	}
	return ""
}

// inDirLex is like inDir but only checks the lexical form of the file names.
// It does not consider symbolic links.
// TODO(rsc): This is a copy of str.HasFilePathPrefix, modified to
// return the suffix. Most uses of str.HasFilePathPrefix should probably
// be calling InDir instead.
func inDirLex(path, dir string) string {
	pv := strings.ToUpper(filepath.VolumeName(path))
	dv := strings.ToUpper(filepath.VolumeName(dir))
	path = path[len(pv):]
	dir = dir[len(dv):]
	switch {
	default:
		return ""
	case pv != dv:
		return ""
	case len(path) == len(dir):
		if path == dir {
			return "."
		}
		return ""
	case dir == "":
		return path
	case len(path) > len(dir):
		if dir[len(dir)-1] == filepath.Separator {
			if path[:len(dir)] == dir {
				return path[len(dir):]
			}
			return ""
		}
		if path[len(dir)] == filepath.Separator && path[:len(dir)] == dir {
			if len(path) == len(dir)+1 {
				return "."
			}
			return path[len(dir)+1:]
		}
		return ""
	}
}
