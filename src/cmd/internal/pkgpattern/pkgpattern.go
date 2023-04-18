// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pkgpattern

import (
	"regexp"
	"strings"
)

// Note: most of this code was originally part of the cmd/go/internal/search
// package; it was migrated here in order to support the use case of
// commands other than cmd/go that need to accept package pattern args.

// TreeCanMatchPattern(pattern)(name) reports whether
// name or children of name can possibly match pattern.
// Pattern is the same limited glob accepted by MatchPattern.
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
	return matchPatternInternal(pattern, true)
}

// MatchSimplePattern returns a function that can be used to check
// whether a given name matches a pattern, where pattern is a limited
// glob pattern in which '...' means 'any string', with no other
// special syntax. There is one special case for MatchPatternSimple:
// according to the rules in "go help packages": a /... at the end of
// the pattern can match an empty string, so that net/... matches both
// net and packages in its subdirectories, like net/http.
func MatchSimplePattern(pattern string) func(name string) bool {
	return matchPatternInternal(pattern, false)
}

func matchPatternInternal(pattern string, vendorExclude bool) func(name string) bool {
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

	if vendorExclude && strings.Contains(pattern, vendorChar) {
		return func(name string) bool { return false }
	}

	re := regexp.QuoteMeta(pattern)
	wild := `.*`
	if vendorExclude {
		wild = `[^` + vendorChar + `]*`
		re = replaceVendor(re, vendorChar)
		switch {
		case strings.HasSuffix(re, `/`+vendorChar+`/\.\.\.`):
			re = strings.TrimSuffix(re, `/`+vendorChar+`/\.\.\.`) + `(/vendor|/` + vendorChar + `/\.\.\.)`
		case re == vendorChar+`/\.\.\.`:
			re = `(/vendor|/` + vendorChar + `/\.\.\.)`
		}
	}
	if strings.HasSuffix(re, `/\.\.\.`) {
		re = strings.TrimSuffix(re, `/\.\.\.`) + `(/\.\.\.)?`
	}
	re = strings.ReplaceAll(re, `\.\.\.`, wild)

	reg := regexp.MustCompile(`^` + re + `$`)

	return func(name string) bool {
		if vendorExclude {
			if strings.Contains(name, vendorChar) {
				return false
			}
			name = replaceVendor(name, vendorChar)
		}
		return reg.MatchString(name)
	}
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
