// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package language

import (
	"sort"
	"strings"
)

// A Builder allows constructing a Tag from individual components.
// Its main user is Compose in the top-level language package.
type Builder struct {
	Tag Tag

	private    string // the x extension
	variants   []string
	extensions []string
}

// Make returns a new Tag from the current settings.
func (b *Builder) Make() Tag {
	t := b.Tag

	if len(b.extensions) > 0 || len(b.variants) > 0 {
		sort.Sort(sortVariants(b.variants))
		sort.Strings(b.extensions)

		if b.private != "" {
			b.extensions = append(b.extensions, b.private)
		}
		n := maxCoreSize + tokenLen(b.variants...) + tokenLen(b.extensions...)
		buf := make([]byte, n)
		p := t.genCoreBytes(buf)
		t.pVariant = byte(p)
		p += appendTokens(buf[p:], b.variants...)
		t.pExt = uint16(p)
		p += appendTokens(buf[p:], b.extensions...)
		t.str = string(buf[:p])
		// We may not always need to remake the string, but when or when not
		// to do so is rather tricky.
		scan := makeScanner(buf[:p])
		t, _ = parse(&scan, "")
		return t

	} else if b.private != "" {
		t.str = b.private
		t.RemakeString()
	}
	return t
}

// SetTag copies all the settings from a given Tag. Any previously set values
// are discarded.
func (b *Builder) SetTag(t Tag) {
	b.Tag.LangID = t.LangID
	b.Tag.RegionID = t.RegionID
	b.Tag.ScriptID = t.ScriptID
	// TODO: optimize
	b.variants = b.variants[:0]
	if variants := t.Variants(); variants != "" {
		for _, vr := range strings.Split(variants[1:], "-") {
			b.variants = append(b.variants, vr)
		}
	}
	b.extensions, b.private = b.extensions[:0], ""
	for _, e := range t.Extensions() {
		b.AddExt(e)
	}
}

// AddExt adds extension e to the tag. e must be a valid extension as returned
// by Tag.Extension. If the extension already exists, it will be discarded,
// except for a -u extension, where non-existing key-type pairs will added.
func (b *Builder) AddExt(e string) {
	if e[0] == 'x' {
		if b.private == "" {
			b.private = e
		}
		return
	}
	for i, s := range b.extensions {
		if s[0] == e[0] {
			if e[0] == 'u' {
				b.extensions[i] += e[1:]
			}
			return
		}
	}
	b.extensions = append(b.extensions, e)
}

// SetExt sets the extension e to the tag. e must be a valid extension as
// returned by Tag.Extension. If the extension already exists, it will be
// overwritten, except for a -u extension, where the individual key-type pairs
// will be set.
func (b *Builder) SetExt(e string) {
	if e[0] == 'x' {
		b.private = e
		return
	}
	for i, s := range b.extensions {
		if s[0] == e[0] {
			if e[0] == 'u' {
				b.extensions[i] = e + s[1:]
			} else {
				b.extensions[i] = e
			}
			return
		}
	}
	b.extensions = append(b.extensions, e)
}

// AddVariant adds any number of variants.
func (b *Builder) AddVariant(v ...string) {
	for _, v := range v {
		if v != "" {
			b.variants = append(b.variants, v)
		}
	}
}

// ClearVariants removes any variants previously added, including those
// copied from a Tag in SetTag.
func (b *Builder) ClearVariants() {
	b.variants = b.variants[:0]
}

// ClearExtensions removes any extensions previously added, including those
// copied from a Tag in SetTag.
func (b *Builder) ClearExtensions() {
	b.private = ""
	b.extensions = b.extensions[:0]
}

func tokenLen(token ...string) (n int) {
	for _, t := range token {
		n += len(t) + 1
	}
	return
}

func appendTokens(b []byte, token ...string) int {
	p := 0
	for _, t := range token {
		b[p] = '-'
		copy(b[p+1:], t)
		p += 1 + len(t)
	}
	return p
}

type sortVariants []string

func (s sortVariants) Len() int {
	return len(s)
}

func (s sortVariants) Swap(i, j int) {
	s[j], s[i] = s[i], s[j]
}

func (s sortVariants) Less(i, j int) bool {
	return variantIndex[s[i]] < variantIndex[s[j]]
}
