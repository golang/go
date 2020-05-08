// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package search

import (
	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"fmt"
	"go/build"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"
)

// A Match represents the result of matching a single package pattern.
type Match struct {
	pattern string   // the pattern itself
	Dirs    []string // if the pattern is local, directories that potentially contain matching packages
	Pkgs    []string // matching packages (import paths)
	Errs    []error  // errors matching the patterns to packages, NOT errors loading those packages

	// Errs may be non-empty even if len(Pkgs) > 0, indicating that some matching
	// packages could be located but results may be incomplete.
	// If len(Pkgs) == 0 && len(Errs) == 0, the pattern is well-formed but did not
	// match any packages.
}

// NewMatch returns a Match describing the given pattern,
// without resolving its packages or errors.
func NewMatch(pattern string) *Match {
	return &Match{pattern: pattern}
}

// Pattern returns the pattern to be matched.
func (m *Match) Pattern() string { return m.pattern }

// AddError appends a MatchError wrapping err to m.Errs.
func (m *Match) AddError(err error) {
	m.Errs = append(m.Errs, &MatchError{Match: m, Err: err})
}

// Literal reports whether the pattern is free of wildcards and meta-patterns.
//
// A literal pattern must match at most one package.
func (m *Match) IsLiteral() bool {
	return !strings.Contains(m.pattern, "...") && !m.IsMeta()
}

// Local reports whether the pattern must be resolved from a specific root or
// directory, such as a filesystem path or a single module.
func (m *Match) IsLocal() bool {
	return build.IsLocalImport(m.pattern) || filepath.IsAbs(m.pattern)
}

// Meta reports whether the pattern is a “meta-package” keyword that represents
// multiple packages, such as "std", "cmd", or "all".
func (m *Match) IsMeta() bool {
	return IsMetaPackage(m.pattern)
}

// IsMetaPackage checks if name is a reserved package name that expands to multiple packages.
func IsMetaPackage(name string) bool {
	return name == "std" || name == "cmd" || name == "all"
}

// A MatchError indicates an error that occurred while attempting to match a
// pattern.
type MatchError struct {
	Match *Match
	Err   error
}

func (e *MatchError) Error() string {
	if e.Match.IsLiteral() {
		return fmt.Sprintf("%s: %v", e.Match.Pattern(), e.Err)
	}
	return fmt.Sprintf("pattern %s: %v", e.Match.Pattern(), e.Err)
}

func (e *MatchError) Unwrap() error {
	return e.Err
}

// MatchPackages sets m.Pkgs to a non-nil slice containing all the packages that
// can be found under the $GOPATH directories and $GOROOT that match the
// pattern. The pattern must be either "all" (all packages), "std" (standard
// packages), "cmd" (standard commands), or a path including "...".
//
// If any errors may have caused the set of packages to be incomplete,
// MatchPackages appends those errors to m.Errs.
func (m *Match) MatchPackages() {
	m.Pkgs = []string{}
	if m.IsLocal() {
		m.AddError(fmt.Errorf("internal error: MatchPackages: %s is not a valid package pattern", m.pattern))
		return
	}

	if m.IsLiteral() {
		m.Pkgs = []string{m.pattern}
		return
	}

	match := func(string) bool { return true }
	treeCanMatch := func(string) bool { return true }
	if !m.IsMeta() {
		match = MatchPattern(m.pattern)
		treeCanMatch = TreeCanMatchPattern(m.pattern)
	}

	have := map[string]bool{
		"builtin": true, // ignore pseudo-package that exists only for documentation
	}
	if !cfg.BuildContext.CgoEnabled {
		have["runtime/cgo"] = true // ignore during walk
	}

	for _, src := range cfg.BuildContext.SrcDirs() {
		if (m.pattern == "std" || m.pattern == "cmd") && src != cfg.GOROOTsrc {
			continue
		}
		src = filepath.Clean(src) + string(filepath.Separator)
		root := src
		if m.pattern == "cmd" {
			root += "cmd" + string(filepath.Separator)
		}
		err := filepath.Walk(root, func(path string, fi os.FileInfo, err error) error {
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
			if m.pattern == "std" && (!IsStandardImportPath(name) || name == "cmd") {
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
					// The package does not actually exist, so record neither the package
					// nor the error.
					return nil
				}
				// There was an error importing path, but not matching it,
				// which is all that Match promises to do.
				// Ignore the import error.
			}

			// If we are expanding "cmd", skip main
			// packages under cmd/vendor. At least as of
			// March, 2017, there is one there for the
			// vendored pprof tool.
			if m.pattern == "cmd" && pkg != nil && strings.HasPrefix(pkg.ImportPath, "cmd/vendor") && pkg.Name == "main" {
				return nil
			}

			m.Pkgs = append(m.Pkgs, name)
			return nil
		})
		if err != nil {
			m.AddError(err)
		}
	}
}

var modRoot string

func SetModRoot(dir string) {
	modRoot = dir
}

// MatchDirs sets m.Dirs to a non-nil slice containing all directories that
// potentially match a local pattern. The pattern must begin with an absolute
// path, or "./", or "../". On Windows, the pattern may use slash or backslash
// separators or a mix of both.
//
// If any errors may have caused the set of directories to be incomplete,
// MatchDirs appends those errors to m.Errs.
func (m *Match) MatchDirs() {
	m.Dirs = []string{}
	if !m.IsLocal() {
		m.AddError(fmt.Errorf("internal error: MatchDirs: %s is not a valid filesystem pattern", m.pattern))
		return
	}

	if m.IsLiteral() {
		m.Dirs = []string{m.pattern}
		return
	}

	// Clean the path and create a matching predicate.
	// filepath.Clean removes "./" prefixes (and ".\" on Windows). We need to
	// preserve these, since they are meaningful in MatchPattern and in
	// returned import paths.
	cleanPattern := filepath.Clean(m.pattern)
	isLocal := strings.HasPrefix(m.pattern, "./") || (os.PathSeparator == '\\' && strings.HasPrefix(m.pattern, `.\`))
	prefix := ""
	if cleanPattern != "." && isLocal {
		prefix = "./"
		cleanPattern = "." + string(os.PathSeparator) + cleanPattern
	}
	slashPattern := filepath.ToSlash(cleanPattern)
	match := MatchPattern(slashPattern)

	// Find directory to begin the scan.
	// Could be smarter but this one optimization
	// is enough for now, since ... is usually at the
	// end of a path.
	i := strings.Index(cleanPattern, "...")
	dir, _ := filepath.Split(cleanPattern[:i])

	// pattern begins with ./ or ../.
	// path.Clean will discard the ./ but not the ../.
	// We need to preserve the ./ for pattern matching
	// and in the returned import paths.

	if modRoot != "" {
		abs, err := filepath.Abs(dir)
		if err != nil {
			m.AddError(err)
			return
		}
		if !hasFilepathPrefix(abs, modRoot) {
			m.AddError(fmt.Errorf("directory %s is outside module root (%s)", abs, modRoot))
			return
		}
	}

	err := filepath.Walk(dir, func(path string, fi os.FileInfo, err error) error {
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
			if fi, err := os.Stat(filepath.Join(path, "go.mod")); err == nil && !fi.IsDir() {
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
			if _, noGo := err.(*build.NoGoError); noGo {
				// The package does not actually exist, so record neither the package
				// nor the error.
				return nil
			}
			// There was an error importing path, but not matching it,
			// which is all that Match promises to do.
			// Ignore the import error.
		}
		m.Dirs = append(m.Dirs, name)
		return nil
	})
	if err != nil {
		m.AddError(err)
	}
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
// Second, any slash-separated pattern element containing a wildcard never
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
		if len(m.Pkgs) == 0 && len(m.Errs) == 0 {
			fmt.Fprintf(os.Stderr, "go: warning: %q matched no packages\n", m.pattern)
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
		m := NewMatch(a)
		if m.IsLocal() {
			m.MatchDirs()

			// Change the file import path to a regular import path if the package
			// is in GOPATH or GOROOT. We don't report errors here; LoadImport
			// (or something similar) will report them later.
			m.Pkgs = make([]string, len(m.Dirs))
			for i, dir := range m.Dirs {
				absDir := dir
				if !filepath.IsAbs(dir) {
					absDir = filepath.Join(base.Cwd, dir)
				}
				if bp, _ := cfg.BuildContext.ImportDir(absDir, build.FindOnly); bp.ImportPath != "" && bp.ImportPath != "." {
					m.Pkgs[i] = bp.ImportPath
				} else {
					m.Pkgs[i] = dir
				}
			}
		} else {
			m.MatchPackages()
		}

		out = append(out, m)
	}
	return out
}

// CleanPatterns returns the patterns to use for the given command line. It
// canonicalizes the patterns but does not evaluate any matches. For patterns
// that are not local or absolute paths, it preserves text after '@' to avoid
// modifying version queries.
func CleanPatterns(patterns []string) []string {
	if len(patterns) == 0 {
		return []string{"."}
	}
	var out []string
	for _, a := range patterns {
		var p, v string
		if build.IsLocalImport(a) || filepath.IsAbs(a) {
			p = a
		} else if i := strings.IndexByte(a, '@'); i < 0 {
			p = a
		} else {
			p = a[:i]
			v = a[i:]
		}

		// Arguments may be either file paths or import paths.
		// As a courtesy to Windows developers, rewrite \ to /
		// in arguments that look like import paths.
		// Don't replace slashes in absolute paths.
		if filepath.IsAbs(p) {
			p = filepath.Clean(p)
		} else {
			if filepath.Separator == '\\' {
				p = strings.ReplaceAll(p, `\`, `/`)
			}

			// Put argument in canonical form, but preserve leading ./.
			if strings.HasPrefix(p, "./") {
				p = "./" + path.Clean(p)
				if p == "./." {
					p = "."
				}
			} else {
				p = path.Clean(p)
			}
		}

		out = append(out, p+v)
	}
	return out
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
