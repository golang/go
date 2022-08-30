// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tests

import (
	"path/filepath"
	"strconv"
	"strings"

	"golang.org/x/tools/go/packages/packagestest"
)

type Normalizer struct {
	path     string
	slashed  string
	escaped  string
	fragment string
}

func CollectNormalizers(exported *packagestest.Exported) []Normalizer {
	// build the path normalizing patterns
	var normalizers []Normalizer
	for _, m := range exported.Modules {
		for fragment := range m.Files {
			n := Normalizer{
				path:     exported.File(m.Name, fragment),
				fragment: fragment,
			}
			if n.slashed = filepath.ToSlash(n.path); n.slashed == n.path {
				n.slashed = ""
			}
			quoted := strconv.Quote(n.path)
			if n.escaped = quoted[1 : len(quoted)-1]; n.escaped == n.path {
				n.escaped = ""
			}
			normalizers = append(normalizers, n)
		}
	}
	return normalizers
}

// NormalizePrefix normalizes a single path at the front of the input string.
func NormalizePrefix(s string, normalizers []Normalizer) string {
	for _, n := range normalizers {
		if t := strings.TrimPrefix(s, n.path); t != s {
			return n.fragment + t
		}
		if t := strings.TrimPrefix(s, n.slashed); t != s {
			return n.fragment + t
		}
		if t := strings.TrimPrefix(s, n.escaped); t != s {
			return n.fragment + t
		}
	}
	return s
}

// Normalize replaces all paths present in s with just the fragment portion
// this is used to make golden files not depend on the temporary paths of the files
func Normalize(s string, normalizers []Normalizer) string {
	type entry struct {
		path     string
		index    int
		fragment string
	}
	var match []entry
	// collect the initial state of all the matchers
	for _, n := range normalizers {
		index := strings.Index(s, n.path)
		if index >= 0 {
			match = append(match, entry{n.path, index, n.fragment})
		}
		if n.slashed != "" {
			index := strings.Index(s, n.slashed)
			if index >= 0 {
				match = append(match, entry{n.slashed, index, n.fragment})
			}
		}
		if n.escaped != "" {
			index := strings.Index(s, n.escaped)
			if index >= 0 {
				match = append(match, entry{n.escaped, index, n.fragment})
			}
		}
	}
	// result should be the same or shorter than the input
	var b strings.Builder
	last := 0
	for {
		// find the nearest path match to the start of the buffer
		next := -1
		nearest := len(s)
		for i, c := range match {
			if c.index >= 0 && nearest > c.index {
				nearest = c.index
				next = i
			}
		}
		// if there are no matches, we copy the rest of the string and are done
		if next < 0 {
			b.WriteString(s[last:])
			return b.String()
		}
		// we have a match
		n := &match[next]
		// copy up to the start of the match
		b.WriteString(s[last:n.index])
		// skip over the filename
		last = n.index + len(n.path)

		// Hack: In multi-module mode, we add a "testmodule/" prefix, so trim
		// it from the fragment.
		fragment := n.fragment
		if strings.HasPrefix(fragment, "testmodule") {
			split := strings.Split(filepath.ToSlash(fragment), "/")
			fragment = filepath.FromSlash(strings.Join(split[1:], "/"))
		}

		// add in the fragment instead
		b.WriteString(fragment)
		// see what the next match for this path is
		n.index = strings.Index(s[last:], n.path)
		if n.index >= 0 {
			n.index += last
		}
	}
}
