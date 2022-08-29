// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Copied from Go distribution src/go/build/build.go, syslist.go.
// That package does not export the ability to process raw file data,
// although we could fake it with an appropriate build.Context
// and a lot of unwrapping.
// More importantly, that package does not implement the tags["*"]
// special case, in which both tag and !tag are considered to be true
// for essentially all tags (except "ignore").
//
// If we added this API to go/build directly, we wouldn't need this
// file anymore, but this API is not terribly general-purpose and we
// don't really want to commit to any public form of it, nor do we
// want to move the core parts of go/build into a top-level internal package.
// These details change very infrequently, so the copy is fine.

package imports

import (
	"bytes"
	"cmd/go/internal/cfg"
	"errors"
	"fmt"
	"go/build/constraint"
	"strings"
	"unicode"
)

var (
	bSlashSlash = []byte("//")
	bStarSlash  = []byte("*/")
	bSlashStar  = []byte("/*")
	bPlusBuild  = []byte("+build")

	goBuildComment = []byte("//go:build")

	errGoBuildWithoutBuild = errors.New("//go:build comment without // +build comment")
	errMultipleGoBuild     = errors.New("multiple //go:build comments")
)

func isGoBuildComment(line []byte) bool {
	if !bytes.HasPrefix(line, goBuildComment) {
		return false
	}
	line = bytes.TrimSpace(line)
	rest := line[len(goBuildComment):]
	return len(rest) == 0 || len(bytes.TrimSpace(rest)) < len(rest)
}

// ShouldBuild reports whether it is okay to use this file,
// The rule is that in the file's leading run of // comments
// and blank lines, which must be followed by a blank line
// (to avoid including a Go package clause doc comment),
// lines beginning with '// +build' are taken as build directives.
//
// The file is accepted only if each such line lists something
// matching the file. For example:
//
//	// +build windows linux
//
// marks the file as applicable only on Windows and Linux.
//
// If tags["*"] is true, then ShouldBuild will consider every
// build tag except "ignore" to be both true and false for
// the purpose of satisfying build tags, in order to estimate
// (conservatively) whether a file could ever possibly be used
// in any build.
func ShouldBuild(content []byte, tags map[string]bool) bool {
	// Identify leading run of // comments and blank lines,
	// which must be followed by a blank line.
	// Also identify any //go:build comments.
	content, goBuild, _, err := parseFileHeader(content)
	if err != nil {
		return false
	}

	// If //go:build line is present, it controls.
	// Otherwise fall back to +build processing.
	var shouldBuild bool
	switch {
	case goBuild != nil:
		x, err := constraint.Parse(string(goBuild))
		if err != nil {
			return false
		}
		shouldBuild = eval(x, tags, true)

	default:
		shouldBuild = true
		p := content
		for len(p) > 0 {
			line := p
			if i := bytes.IndexByte(line, '\n'); i >= 0 {
				line, p = line[:i], p[i+1:]
			} else {
				p = p[len(p):]
			}
			line = bytes.TrimSpace(line)
			if !bytes.HasPrefix(line, bSlashSlash) || !bytes.Contains(line, bPlusBuild) {
				continue
			}
			text := string(line)
			if !constraint.IsPlusBuild(text) {
				continue
			}
			if x, err := constraint.Parse(text); err == nil {
				if !eval(x, tags, true) {
					shouldBuild = false
				}
			}
		}
	}

	return shouldBuild
}

func parseFileHeader(content []byte) (trimmed, goBuild []byte, sawBinaryOnly bool, err error) {
	end := 0
	p := content
	ended := false       // found non-blank, non-// line, so stopped accepting // +build lines
	inSlashStar := false // in /* */ comment

Lines:
	for len(p) > 0 {
		line := p
		if i := bytes.IndexByte(line, '\n'); i >= 0 {
			line, p = line[:i], p[i+1:]
		} else {
			p = p[len(p):]
		}
		line = bytes.TrimSpace(line)
		if len(line) == 0 && !ended { // Blank line
			// Remember position of most recent blank line.
			// When we find the first non-blank, non-// line,
			// this "end" position marks the latest file position
			// where a // +build line can appear.
			// (It must appear _before_ a blank line before the non-blank, non-// line.
			// Yes, that's confusing, which is part of why we moved to //go:build lines.)
			// Note that ended==false here means that inSlashStar==false,
			// since seeing a /* would have set ended==true.
			end = len(content) - len(p)
			continue Lines
		}
		if !bytes.HasPrefix(line, bSlashSlash) { // Not comment line
			ended = true
		}

		if !inSlashStar && isGoBuildComment(line) {
			if goBuild != nil {
				return nil, nil, false, errMultipleGoBuild
			}
			goBuild = line
		}

	Comments:
		for len(line) > 0 {
			if inSlashStar {
				if i := bytes.Index(line, bStarSlash); i >= 0 {
					inSlashStar = false
					line = bytes.TrimSpace(line[i+len(bStarSlash):])
					continue Comments
				}
				continue Lines
			}
			if bytes.HasPrefix(line, bSlashSlash) {
				continue Lines
			}
			if bytes.HasPrefix(line, bSlashStar) {
				inSlashStar = true
				line = bytes.TrimSpace(line[len(bSlashStar):])
				continue Comments
			}
			// Found non-comment text.
			break Lines
		}
	}

	return content[:end], goBuild, sawBinaryOnly, nil
}

// matchTag reports whether the tag name is valid and tags[name] is true.
// As a special case, if tags["*"] is true and name is not empty or ignore,
// then matchTag will return prefer instead of the actual answer,
// which allows the caller to pretend in that case that most tags are
// both true and false.
func matchTag(name string, tags map[string]bool, prefer bool) bool {
	// Tags must be letters, digits, underscores or dots.
	// Unlike in Go identifiers, all digits are fine (e.g., "386").
	for _, c := range name {
		if !unicode.IsLetter(c) && !unicode.IsDigit(c) && c != '_' && c != '.' {
			return false
		}
	}

	if tags["*"] && name != "" && name != "ignore" {
		// Special case for gathering all possible imports:
		// if we put * in the tags map then all tags
		// except "ignore" are considered both present and not
		// (so we return true no matter how 'want' is set).
		return prefer
	}

	if tags[name] {
		return true
	}

	switch name {
	case "linux":
		return tags["android"]
	case "solaris":
		return tags["illumos"]
	case "darwin":
		return tags["ios"]
	case "unix":
		return unixOS[cfg.BuildContext.GOOS]
	default:
		return false
	}
}

// eval is like
//
//	x.Eval(func(tag string) bool { return matchTag(tag, tags) })
//
// except that it implements the special case for tags["*"] meaning
// all tags are both true and false at the same time.
func eval(x constraint.Expr, tags map[string]bool, prefer bool) bool {
	switch x := x.(type) {
	case *constraint.TagExpr:
		return matchTag(x.Tag, tags, prefer)
	case *constraint.NotExpr:
		return !eval(x.X, tags, !prefer)
	case *constraint.AndExpr:
		return eval(x.X, tags, prefer) && eval(x.Y, tags, prefer)
	case *constraint.OrExpr:
		return eval(x.X, tags, prefer) || eval(x.Y, tags, prefer)
	}
	panic(fmt.Sprintf("unexpected constraint expression %T", x))
}

// Eval is like
//
//	x.Eval(func(tag string) bool { return matchTag(tag, tags) })
//
// except that it implements the special case for tags["*"] meaning
// all tags are both true and false at the same time.
func Eval(x constraint.Expr, tags map[string]bool, prefer bool) bool {
	return eval(x, tags, prefer)
}

// MatchFile returns false if the name contains a $GOOS or $GOARCH
// suffix which does not match the current system.
// The recognized name formats are:
//
//	name_$(GOOS).*
//	name_$(GOARCH).*
//	name_$(GOOS)_$(GOARCH).*
//	name_$(GOOS)_test.*
//	name_$(GOARCH)_test.*
//	name_$(GOOS)_$(GOARCH)_test.*
//
// Exceptions:
//
//	if GOOS=android, then files with GOOS=linux are also matched.
//	if GOOS=illumos, then files with GOOS=solaris are also matched.
//	if GOOS=ios, then files with GOOS=darwin are also matched.
//
// If tags["*"] is true, then MatchFile will consider all possible
// GOOS and GOARCH to be available and will consequently
// always return true.
func MatchFile(name string, tags map[string]bool) bool {
	if tags["*"] {
		return true
	}
	if dot := strings.Index(name, "."); dot != -1 {
		name = name[:dot]
	}

	// Before Go 1.4, a file called "linux.go" would be equivalent to having a
	// build tag "linux" in that file. For Go 1.4 and beyond, we require this
	// auto-tagging to apply only to files with a non-empty prefix, so
	// "foo_linux.go" is tagged but "linux.go" is not. This allows new operating
	// systems, such as android, to arrive without breaking existing code with
	// innocuous source code in "android.go". The easiest fix: cut everything
	// in the name before the initial _.
	i := strings.Index(name, "_")
	if i < 0 {
		return true
	}
	name = name[i:] // ignore everything before first _

	l := strings.Split(name, "_")
	if n := len(l); n > 0 && l[n-1] == "test" {
		l = l[:n-1]
	}
	n := len(l)
	if n >= 2 && KnownOS[l[n-2]] && KnownArch[l[n-1]] {
		return matchTag(l[n-2], tags, true) && matchTag(l[n-1], tags, true)
	}
	if n >= 1 && KnownOS[l[n-1]] {
		return matchTag(l[n-1], tags, true)
	}
	if n >= 1 && KnownArch[l[n-1]] {
		return matchTag(l[n-1], tags, true)
	}
	return true
}

var KnownOS = map[string]bool{
	"aix":       true,
	"android":   true,
	"darwin":    true,
	"dragonfly": true,
	"freebsd":   true,
	"hurd":      true,
	"illumos":   true,
	"ios":       true,
	"js":        true,
	"linux":     true,
	"nacl":      true, // legacy; don't remove
	"netbsd":    true,
	"openbsd":   true,
	"plan9":     true,
	"solaris":   true,
	"windows":   true,
	"zos":       true,
}

// unixOS is the set of GOOS values matched by the "unix" build tag.
// This is not used for filename matching.
// This is the same list as in go/build/syslist.go and cmd/dist/build.go.
var unixOS = map[string]bool{
	"aix":       true,
	"android":   true,
	"darwin":    true,
	"dragonfly": true,
	"freebsd":   true,
	"hurd":      true,
	"illumos":   true,
	"ios":       true,
	"linux":     true,
	"netbsd":    true,
	"openbsd":   true,
	"solaris":   true,
}

var KnownArch = map[string]bool{
	"386":         true,
	"amd64":       true,
	"amd64p32":    true, // legacy; don't remove
	"arm":         true,
	"armbe":       true,
	"arm64":       true,
	"arm64be":     true,
	"ppc64":       true,
	"ppc64le":     true,
	"mips":        true,
	"mipsle":      true,
	"mips64":      true,
	"mips64le":    true,
	"mips64p32":   true,
	"mips64p32le": true,
	"loong64":     true,
	"ppc":         true,
	"riscv":       true,
	"riscv64":     true,
	"s390":        true,
	"s390x":       true,
	"sparc":       true,
	"sparc64":     true,
	"wasm":        true,
}
