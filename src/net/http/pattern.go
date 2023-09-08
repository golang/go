// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Patterns for ServeMux routing.

package http

import (
	"errors"
	"fmt"
	"strings"
	"unicode"
)

// A pattern is something that can be matched against an HTTP request.
// It has an optional method, an optional host, and a path.
type pattern struct {
	str    string // original string
	method string
	host   string
	// The representation of a path differs from the surface syntax, which
	// simplifies most algorithms.
	//
	// Paths ending in '/' are represented with an anonymous "..." wildcard.
	// For example, the path "a/" is represented as a literal segment "a" followed
	// by a segment with multi==true.
	//
	// Paths ending in "{$}" are represented with the literal segment "/".
	// For example, the path "a/{$}" is represented as a literal segment "a" followed
	// by a literal segment "/".
	segments []segment
	loc      string // source location of registering call, for helpful messages
}

func (p *pattern) String() string { return p.str }

func (p *pattern) lastSegment() segment {
	return p.segments[len(p.segments)-1]
}

// A segment is a pattern piece that matches one or more path segments, or
// a trailing slash.
//
// If wild is false, it matches a literal segment, or, if s == "/", a trailing slash.
// Examples:
//
//	"a" => segment{s: "a"}
//	"/{$}" => segment{s: "/"}
//
// If wild is true and multi is false, it matches a single path segment.
// Example:
//
//	"{x}" => segment{s: "x", wild: true}
//
// If both wild and multi are true, it matches all remaining path segments.
// Example:
//
//	"{rest...}" => segment{s: "rest", wild: true, multi: true}
type segment struct {
	s     string // literal or wildcard name or "/" for "/{$}".
	wild  bool
	multi bool // "..." wildcard
}

// parsePattern parses a string into a Pattern.
// The string's syntax is
//
//	[METHOD] [HOST]/[PATH]
//
// where:
//   - METHOD is an HTTP method
//   - HOST is a hostname
//   - PATH consists of slash-separated segments, where each segment is either
//     a literal or a wildcard of the form "{name}", "{name...}", or "{$}".
//
// METHOD, HOST and PATH are all optional; that is, the string can be "/".
// If METHOD is present, it must be followed by a single space.
// Wildcard names must be valid Go identifiers.
// The "{$}" and "{name...}" wildcard must occur at the end of PATH.
// PATH may end with a '/'.
// Wildcard names in a path must be distinct.
func parsePattern(s string) (*pattern, error) {
	if len(s) == 0 {
		return nil, errors.New("empty pattern")
	}
	// TODO(jba): record the rune offset in s to provide more information in errors.
	method, rest, found := strings.Cut(s, " ")
	if !found {
		rest = method
		method = ""
	}
	if method != "" && !validMethod(method) {
		return nil, fmt.Errorf("net/http: invalid method %q", method)
	}
	p := &pattern{str: s, method: method}

	i := strings.IndexByte(rest, '/')
	if i < 0 {
		return nil, errors.New("host/path missing /")
	}
	p.host = rest[:i]
	rest = rest[i:]
	if strings.IndexByte(p.host, '{') >= 0 {
		return nil, errors.New("host contains '{' (missing initial '/'?)")
	}
	// At this point, rest is the path.

	// An unclean path with a method that is not CONNECT can never match,
	// because paths are cleaned before matching.
	if method != "" && method != "CONNECT" && rest != cleanPath(rest) {
		return nil, errors.New("non-CONNECT pattern with unclean path can never match")
	}

	seenNames := map[string]bool{} // remember wildcard names to catch dups
	for len(rest) > 0 {
		// Invariant: rest[0] == '/'.
		rest = rest[1:]
		if len(rest) == 0 {
			// Trailing slash.
			p.segments = append(p.segments, segment{wild: true, multi: true})
			break
		}
		i := strings.IndexByte(rest, '/')
		if i < 0 {
			i = len(rest)
		}
		var seg string
		seg, rest = rest[:i], rest[i:]
		if i := strings.IndexByte(seg, '{'); i < 0 {
			// Literal.
			p.segments = append(p.segments, segment{s: seg})
		} else {
			// Wildcard.
			if i != 0 {
				return nil, errors.New("bad wildcard segment (must start with '{')")
			}
			if seg[len(seg)-1] != '}' {
				return nil, errors.New("bad wildcard segment (must end with '}')")
			}
			name := seg[1 : len(seg)-1]
			if name == "$" {
				if len(rest) != 0 {
					return nil, errors.New("{$} not at end")
				}
				p.segments = append(p.segments, segment{s: "/"})
				break
			}
			name, multi := strings.CutSuffix(name, "...")
			if multi && len(rest) != 0 {
				return nil, errors.New("{...} wildcard not at end")
			}
			if name == "" {
				return nil, errors.New("empty wildcard")
			}
			if !isValidWildcardName(name) {
				return nil, fmt.Errorf("bad wildcard name %q", name)
			}
			if seenNames[name] {
				return nil, fmt.Errorf("duplicate wildcard name %q", name)
			}
			seenNames[name] = true
			p.segments = append(p.segments, segment{s: name, wild: true, multi: multi})
		}
	}
	return p, nil
}

func isValidHTTPToken(s string) bool {
	if s == "" {
		return false
	}
	// See https://www.rfc-editor.org/rfc/rfc9110#section-5.6.2.
	for _, r := range s {
		if !unicode.IsLetter(r) && !unicode.IsDigit(r) && !strings.ContainsRune("!#$%&'*+.^_`|~-", r) {
			return false
		}
	}
	return true
}

func isValidWildcardName(s string) bool {
	if s == "" {
		return false
	}
	// Valid Go identifier.
	for i, c := range s {
		if !unicode.IsLetter(c) && c != '_' && (i == 0 || !unicode.IsDigit(c)) {
			return false
		}
	}
	return true
}

// relationship is a relationship between two patterns, p1 and p2.
type relationship string

const (
	equivalent   relationship = "equivalent"   // both match the same requests
	moreGeneral  relationship = "moreGeneral"  // p1 matches everything p2 does & more
	moreSpecific relationship = "moreSpecific" // p2 matches everything p1 does & more
	disjoint     relationship = "disjoint"     // there is no request that both match
	overlaps     relationship = "overlaps"     // there is a request that both match, but neither is more specific
)

// conflictsWith reports whether p1 conflicts with p2, that is, whether
// there is a request that both match but where neither is higher precedence
// than the other.
//
//	Precedence is defined by two rules:
//	1. Patterns with a host win over patterns without a host.
//	2. Patterns whose method and path is more specific win. One pattern is more
//	   specific than another if the second matches all the (method, path) pairs
//	   of the first and more.
//
// If rule 1 doesn't apply, then two patterns conflict if their relationship
// is either equivalence (they match the same set of requests) or overlap
// (they both match some requests, but neither is more specific than the other).
func (p1 *pattern) conflictsWith(p2 *pattern) bool {
	if p1.host != p2.host {
		// Either one host is empty and the other isn't, in which case the
		// one with the host wins by rule 1, or neither host is empty
		// and they differ, so they won't match the same paths.
		return false
	}
	rel := p1.comparePathsAndMethods(p2)
	return rel == equivalent || rel == overlaps
}

func (p1 *pattern) comparePathsAndMethods(p2 *pattern) relationship {
	mrel := p1.compareMethods(p2)
	// Optimization: avoid a call to comparePaths.
	if mrel == disjoint {
		return disjoint
	}
	prel := p1.comparePaths(p2)
	return combineRelationships(mrel, prel)
}

// compareMethods determines the relationship between the method
// part of patterns p1 and p2.
//
// A method can either be empty, "GET", or something else.
// The empty string matches any method, so it is the most general.
// "GET" matches both GET and HEAD.
// Anything else matches only itself.
func (p1 *pattern) compareMethods(p2 *pattern) relationship {
	if p1.method == p2.method {
		return equivalent
	}
	if p1.method == "" {
		// p1 matches any method, but p2 does not, so p1 is more general.
		return moreGeneral
	}
	if p2.method == "" {
		return moreSpecific
	}
	if p1.method == "GET" && p2.method == "HEAD" {
		// p1 matches GET and HEAD; p2 matches only HEAD.
		return moreGeneral
	}
	if p2.method == "GET" && p1.method == "HEAD" {
		return moreSpecific
	}
	return disjoint
}

// comparePaths determines the relationship between the path
// part of two patterns.
func (p1 *pattern) comparePaths(p2 *pattern) relationship {
	// Optimization: if a path pattern doesn't end in a multi ("...") wildcard, then it
	// can only match paths with the same number of segments.
	if len(p1.segments) != len(p2.segments) && !p1.lastSegment().multi && !p2.lastSegment().multi {
		return disjoint
	}
	var segs1, segs2 []segment
	// Look at corresponding segments in the two path patterns.
	rel := equivalent
	for segs1, segs2 = p1.segments, p2.segments; len(segs1) > 0 && len(segs2) > 0; segs1, segs2 = segs1[1:], segs2[1:] {
		rel = combineRelationships(rel, compareSegments(segs1[0], segs2[0]))
		if rel == disjoint || rel == overlaps {
			return rel
		}
	}
	// We've reached the end of the corresponding segments of the patterns.
	// If they have the same number of segments, then we've already determined
	// their relationship.
	if len(segs1) == 0 && len(segs2) == 0 {
		return rel
	}
	// Otherwise, the only way they could fail to be disjoint is if the shorter
	// pattern ends in a multi and is more general.
	if len(segs1) < len(segs2) && p1.lastSegment().multi && rel == moreGeneral {
		return moreGeneral
	}
	if len(segs2) < len(segs1) && p2.lastSegment().multi && rel == moreSpecific {
		return moreSpecific
	}
	return disjoint
}

// compareSegments determines the relationship between two segments.
func compareSegments(s1, s2 segment) relationship {
	if s1.multi && s2.multi {
		return equivalent
	}
	if s1.multi {
		return moreGeneral
	}
	if s2.multi {
		return moreSpecific
	}
	if s1.wild && s2.wild {
		return equivalent
	}
	if s1.wild {
		if s2.s == "/" {
			// A single wildcard doesn't match a trailing slash.
			return disjoint
		}
		return moreGeneral
	}
	if s2.wild {
		if s1.s == "/" {
			return disjoint
		}
		return moreSpecific
	}
	// Both literals.
	if s1.s == s2.s {
		return equivalent
	}
	return disjoint
}

// combineRelationships determines the overall relationship of two patterns
// given the relationships of a partition of the patterns into two parts.
//
// For example, if p1 is more general than p2 in one way but equivalent
// in the other, then it is more general overall.
//
// Or if p1 is more general in one way and more specific in the other, then
// they overlap.
func combineRelationships(r1, r2 relationship) relationship {
	switch r1 {
	case equivalent:
		return r2
	case disjoint, overlaps:
		return r1
	case moreGeneral, moreSpecific:
		switch r2 {
		case equivalent:
			return r1
		case inverseRelationship(r1):
			return overlaps
		default:
			return r2
		}
	default:
		panic(fmt.Sprintf("unknown relationship %q", r1))
	}
}

// If p1 has relationship `r` to p2, then
// p2 has inverseRelationship(r) to p1.
func inverseRelationship(r relationship) relationship {
	switch r {
	case moreSpecific:
		return moreGeneral
	case moreGeneral:
		return moreSpecific
	default:
		return r
	}
}
