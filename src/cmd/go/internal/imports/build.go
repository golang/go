// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Copied from Go distribution src/go/build/build.go, syslist.go

package imports

import (
	"bytes"
	"strings"
	"unicode"
)

var slashslash = []byte("//")

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
//
func ShouldBuild(content []byte, tags map[string]bool) bool {
	// Pass 1. Identify leading run of // comments and blank lines,
	// which must be followed by a blank line.
	end := 0
	p := content
	for len(p) > 0 {
		line := p
		if i := bytes.IndexByte(line, '\n'); i >= 0 {
			line, p = line[:i], p[i+1:]
		} else {
			p = p[len(p):]
		}
		line = bytes.TrimSpace(line)
		if len(line) == 0 { // Blank line
			end = len(content) - len(p)
			continue
		}
		if !bytes.HasPrefix(line, slashslash) { // Not comment line
			break
		}
	}
	content = content[:end]

	// Pass 2.  Process each line in the run.
	p = content
	allok := true
	for len(p) > 0 {
		line := p
		if i := bytes.IndexByte(line, '\n'); i >= 0 {
			line, p = line[:i], p[i+1:]
		} else {
			p = p[len(p):]
		}
		line = bytes.TrimSpace(line)
		if !bytes.HasPrefix(line, slashslash) {
			continue
		}
		line = bytes.TrimSpace(line[len(slashslash):])
		if len(line) > 0 && line[0] == '+' {
			// Looks like a comment +line.
			f := strings.Fields(string(line))
			if f[0] == "+build" {
				ok := false
				for _, tok := range f[1:] {
					if matchTags(tok, tags) {
						ok = true
					}
				}
				if !ok {
					allok = false
				}
			}
		}
	}

	return allok
}

// matchTags reports whether the name is one of:
//
//	tag (if tags[tag] is true)
//	!tag (if tags[tag] is false)
//	a comma-separated list of any of these
//
func matchTags(name string, tags map[string]bool) bool {
	if name == "" {
		return false
	}
	if i := strings.Index(name, ","); i >= 0 {
		// comma-separated list
		ok1 := matchTags(name[:i], tags)
		ok2 := matchTags(name[i+1:], tags)
		return ok1 && ok2
	}
	if strings.HasPrefix(name, "!!") { // bad syntax, reject always
		return false
	}
	if strings.HasPrefix(name, "!") { // negation
		return len(name) > 1 && matchTag(name[1:], tags, false)
	}
	return matchTag(name, tags, true)
}

// matchTag reports whether the tag name is valid and satisfied by tags[name]==want.
func matchTag(name string, tags map[string]bool, want bool) bool {
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
		return true
	}

	have := tags[name]
	if name == "linux" {
		have = have || tags["android"]
	}
	if name == "solaris" {
		have = have || tags["illumos"]
	}
	return have == want
}

// MatchFile returns false if the name contains a $GOOS or $GOARCH
// suffix which does not match the current system.
// The recognized name formats are:
//
//     name_$(GOOS).*
//     name_$(GOARCH).*
//     name_$(GOOS)_$(GOARCH).*
//     name_$(GOOS)_test.*
//     name_$(GOARCH)_test.*
//     name_$(GOOS)_$(GOARCH)_test.*
//
// Exceptions:
//     if GOOS=android, then files with GOOS=linux are also matched.
//     if GOOS=illumos, then files with GOOS=solaris are also matched.
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
	"js":        true,
	"linux":     true,
	"nacl":      true,
	"netbsd":    true,
	"openbsd":   true,
	"plan9":     true,
	"solaris":   true,
	"windows":   true,
	"zos":       true,
}

var KnownArch = map[string]bool{
	"386":         true,
	"amd64":       true,
	"amd64p32":    true,
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
	"ppc":         true,
	"riscv":       true,
	"riscv64":     true,
	"s390":        true,
	"s390x":       true,
	"sparc":       true,
	"sparc64":     true,
	"wasm":        true,
}
