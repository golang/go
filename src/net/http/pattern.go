// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Patterns for ServeMux routing.

package http

import (
	"errors"
	"fmt"
	"net/url"
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
// If METHOD is present, it must be followed by at least one ' ' or '\t'.
// Wildcard names must be valid Go identifiers.
// The "{$}" and "{name...}" wildcard must occur at the end of PATH.
// PATH may end with a '/'.
// Wildcard names in a path must be distinct.
func parsePattern(s string) (_ *pattern, err error) {
	if len(s) == 0 {
		return nil, errors.New("empty pattern")
	}
	off := 0 // offset into string
	defer func() {
		if err != nil {
			err = fmt.Errorf("at offset %d: %w", off, err)
		}
	}()

	method, rest, found := s, "", false
	if i := strings.IndexAny(s, " \t"); i >= 0 {
		method, rest, found = s[:i], strings.TrimSpace(s[i+1:]), true
	}
	if !found {
		rest = method
		method = ""
	}
	if method != "" && !validMethod(method) {
		return nil, fmt.Errorf("invalid method %q", method)
	}
	p := &pattern{str: s, method: method}

	if found {
		off = len(method) + 1
	}
	i := strings.IndexByte(rest, '/')
	if i < 0 {
		return nil, errors.New("host/path missing /")
	}
	p.host = rest[:i]
	rest = rest[i:]
	if j := strings.IndexByte(p.host, '{'); j >= 0 {
		off += j
		return nil, errors.New("host contains '{' (missing initial '/'?)")
	}
	// At this point, rest is the path.
	off += i

	// An unclean path with a method that is not CONNECT can never match,
	// because paths are cleaned before matching.
	if method != "" && method != "CONNECT" && rest != cleanPath(rest) {
		return nil, errors.New("non-CONNECT pattern with unclean path can never match")
	}

	seenNames := map[string]bool{} // remember wildcard names to catch dups
	for len(rest) > 0 {
		// Invariant: rest[0] == '/'.
		rest = rest[1:]
		off = len(s) - len(rest)
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
			seg = pathUnescape(seg)
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

func pathUnescape(path string) string {
	u, err := url.PathUnescape(path)
	if err != nil {
		// Invalidly escaped path; use the original
		return path
	}
	return u
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

	// Consider corresponding segments in the two path patterns.
	var segs1, segs2 []segment
	rel := equivalent
	for segs1, segs2 = p1.segments, p2.segments; len(segs1) > 0 && len(segs2) > 0; segs1, segs2 = segs1[1:], segs2[1:] {
		rel = combineRelationships(rel, compareSegments(segs1[0], segs2[0]))
		if rel == disjoint {
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
	// pattern ends in a multi. In that case, that multi is more general
	// than the remainder of the longer pattern, so combine those two relationships.
	if len(segs1) < len(segs2) && p1.lastSegment().multi {
		return combineRelationships(rel, moreGeneral)
	}
	if len(segs2) < len(segs1) && p2.lastSegment().multi {
		return combineRelationships(rel, moreSpecific)
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
	case disjoint:
		return disjoint
	case overlaps:
		if r2 == disjoint {
			return disjoint
		}
		return overlaps
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

// isLitOrSingle reports whether the segment is a non-dollar literal or a single wildcard.
func isLitOrSingle(seg segment) bool {
	if seg.wild {
		return !seg.multi
	}
	return seg.s != "/"
}

// describeConflict returns an explanation of why two patterns conflict.
func describeConflict(p1, p2 *pattern) string {
	mrel := p1.compareMethods(p2)
	prel := p1.comparePaths(p2)
	rel := combineRelationships(mrel, prel)
	if rel == equivalent {
		return fmt.Sprintf("%s matches the same requests as %s", p1, p2)
	}
	if rel != overlaps {
		panic("describeConflict called with non-conflicting patterns")
	}
	if prel == overlaps {
		return fmt.Sprintf(`%[1]s and %[2]s both match some paths, like %[3]q.
But neither is more specific than the other.
%[1]s matches %[4]q, but %[2]s doesn't.
%[2]s matches %[5]q, but %[1]s doesn't.`,
			p1, p2, commonPath(p1, p2), differencePath(p1, p2), differencePath(p2, p1))
	}
	if mrel == moreGeneral && prel == moreSpecific {
		return fmt.Sprintf("%s matches more methods than %s, but has a more specific path pattern", p1, p2)
	}
	if mrel == moreSpecific && prel == moreGeneral {
		return fmt.Sprintf("%s matches fewer methods than %s, but has a more general path pattern", p1, p2)
	}
	return fmt.Sprintf("bug: unexpected way for two patterns %s and %s to conflict: methods %s, paths %s", p1, p2, mrel, prel)
}

// writeMatchingPath writes to b a path that matches the segments.
func writeMatchingPath(b *strings.Builder, segs []segment) {
	for _, s := range segs {
		writeSegment(b, s)
	}
}

func writeSegment(b *strings.Builder, s segment) {
	b.WriteByte('/')
	if !s.multi && s.s != "/" {
		b.WriteString(s.s)
	}
}

// commonPath returns a path that both p1 and p2 match.
// It assumes there is such a path.
func commonPath(p1, p2 *pattern) string {
	var b strings.Builder
	var segs1, segs2 []segment
	for segs1, segs2 = p1.segments, p2.segments; len(segs1) > 0 && len(segs2) > 0; segs1, segs2 = segs1[1:], segs2[1:] {
		if s1 := segs1[0]; s1.wild {
			writeSegment(&b, segs2[0])
		} else {
			writeSegment(&b, s1)
		}
	}
	if len(segs1) > 0 {
		writeMatchingPath(&b, segs1)
	} else if len(segs2) > 0 {
		writeMatchingPath(&b, segs2)
	}
	return b.String()
}

// differencePath returns a path that p1 matches and p2 doesn't.
// It assumes there is such a path.
func differencePath(p1, p2 *pattern) string {
	var b strings.Builder

	var segs1, segs2 []segment
	for segs1, segs2 = p1.segments, p2.segments; len(segs1) > 0 && len(segs2) > 0; segs1, segs2 = segs1[1:], segs2[1:] {
		s1 := segs1[0]
		s2 := segs2[0]
		if s1.multi && s2.multi {
			// From here the patterns match the same paths, so we must have found a difference earlier.
			b.WriteByte('/')
			return b.String()

		}
		if s1.multi && !s2.multi {
			// s1 ends in a "..." wildcard but s2 does not.
			// A trailing slash will distinguish them, unless s2 ends in "{$}",
			// in which case any segment will do; prefer the wildcard name if
			// it has one.
			b.WriteByte('/')
			if s2.s == "/" {
				if s1.s != "" {
					b.WriteString(s1.s)
				} else {
					b.WriteString("x")
				}
			}
			return b.String()
		}
		if !s1.multi && s2.multi {
			writeSegment(&b, s1)
		} else if s1.wild && s2.wild {
			// Both patterns will match whatever we put here; use
			// the first wildcard name.
			writeSegment(&b, s1)
		} else if s1.wild && !s2.wild {
			// s1 is a wildcard, s2 is a literal.
			// Any segment other than s2.s will work.
			// Prefer the wildcard name, but if it's the same as the literal,
			// tweak the literal.
			if s1.s != s2.s {
				writeSegment(&b, s1)
			} else {
				b.WriteByte('/')
				b.WriteString(s2.s + "x")
			}
		} else if !s1.wild && s2.wild {
			writeSegment(&b, s1)
		} else {
			// Both are literals. A precondition of this function is that the
			// patterns overlap, so they must be the same literal. Use it.
			if s1.s != s2.s {
				panic(fmt.Sprintf("literals differ: %q and %q", s1.s, s2.s))
			}
			writeSegment(&b, s1)
		}
	}
	if len(segs1) > 0 {
		// p1 is longer than p2, and p2 does not end in a multi.
		// Anything that matches the rest of p1 will do.
		writeMatchingPath(&b, segs1)
	} else if len(segs2) > 0 {
		writeMatchingPath(&b, segs2)
	}
	return b.String()
}
