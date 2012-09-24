// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"exp/norm"
	"math"
	"math/rand"
	"strings"
	"unicode"
	"unicode/utf16"
	"unicode/utf8"
)

// parent computes the parent locale for the given locale.
// It returns false if the parent is already root.
func parent(locale string) (parent string, ok bool) {
	if locale == "root" {
		return "", false
	}
	if i := strings.LastIndex(locale, "_"); i != -1 {
		return locale[:i], true
	}
	return "root", true
}

// rewriter is used to both unique strings and create variants of strings
// to add to the test set.
type rewriter struct {
	seen     map[string]bool
	addCases bool
}

func newRewriter() *rewriter {
	return &rewriter{
		seen: make(map[string]bool),
	}
}

func (r *rewriter) insert(a []string, s string) []string {
	if !r.seen[s] {
		r.seen[s] = true
		a = append(a, s)
	}
	return a
}

// rewrite takes a sequence of strings in, adds variants of the these strings
// based on options and removes duplicates.
func (r *rewriter) rewrite(ss []string) []string {
	ns := []string{}
	for _, s := range ss {
		ns = r.insert(ns, s)
		if r.addCases {
			rs := []rune(s)
			rn := rs[0]
			for c := unicode.SimpleFold(rn); c != rn; c = unicode.SimpleFold(c) {
				rs[0] = c
				ns = r.insert(ns, string(rs))
			}
		}
	}
	return ns
}

// exemplarySet holds a parsed set of characters from the exemplarCharacters table.
type exemplarySet struct {
	typ       exemplarType
	set       []string
	charIndex int // cumulative total of phrases, including this set
}

type phraseGenerator struct {
	sets [exN]exemplarySet
	n    int
}

func (g *phraseGenerator) init(locale string) {
	ec := exemplarCharacters
	// get sets for locale or parent locale if the set is not defined.
	for i := range g.sets {
		for p, ok := locale, true; ok; p, ok = parent(p) {
			if set, ok := ec[p]; ok && set[i] != "" {
				g.sets[i].set = strings.Split(set[i], " ")
				break
			}
		}
	}
	r := newRewriter()
	r.addCases = *cases
	for i := range g.sets {
		g.sets[i].set = r.rewrite(g.sets[i].set)
	}
	// compute indexes
	for i, set := range g.sets {
		g.n += len(set.set)
		g.sets[i].charIndex = g.n
	}
}

// phrase returns the ith phrase, where i < g.n.
func (g *phraseGenerator) phrase(i int) string {
	for _, set := range g.sets {
		if i < set.charIndex {
			return set.set[i-(set.charIndex-len(set.set))]
		}
	}
	panic("index out of range")
}

// generate generates inputs by combining all pairs of examplar strings.
// If doNorm is true, all input strings are normalized to NFC.
// TODO: allow other variations, statistical models, and random
// trailing sequences.
func (g *phraseGenerator) generate(doNorm bool) []Input {
	const (
		M         = 1024 * 1024
		buf8Size  = 30 * M
		buf16Size = 10 * M
	)
	// TODO: use a better way to limit the input size.
	if sq := int(math.Sqrt(float64(*limit))); g.n > sq {
		g.n = sq
	}
	size := g.n * g.n
	a := make([]Input, 0, size)
	buf8 := make([]byte, 0, buf8Size)
	buf16 := make([]uint16, 0, buf16Size)

	addInput := func(str string) {
		buf8 = buf8[len(buf8):]
		buf16 = buf16[len(buf16):]
		if len(str) > cap(buf8) {
			buf8 = make([]byte, 0, buf8Size)
		}
		if len(str) > cap(buf16) {
			buf16 = make([]uint16, 0, buf16Size)
		}
		if doNorm {
			buf8 = norm.NFC.AppendString(buf8, str)
		} else {
			buf8 = append(buf8, str...)
		}
		buf16 = appendUTF16(buf16, buf8)
		a = append(a, makeInput(buf8, buf16))
	}
	for i := 0; i < g.n; i++ {
		p1 := g.phrase(i)
		addInput(p1)
		for j := 0; j < g.n; j++ {
			p2 := g.phrase(j)
			addInput(p1 + p2)
		}
	}
	// permutate
	rnd := rand.New(rand.NewSource(int64(rand.Int())))
	for i := range a {
		j := i + rnd.Intn(len(a)-i)
		a[i], a[j] = a[j], a[i]
		a[i].index = i // allow restoring this order if input is used multiple times.
	}
	return a
}

func appendUTF16(buf []uint16, s []byte) []uint16 {
	for len(s) > 0 {
		r, sz := utf8.DecodeRune(s)
		s = s[sz:]
		r1, r2 := utf16.EncodeRune(r)
		if r1 != 0xFFFD {
			buf = append(buf, uint16(r1), uint16(r2))
		} else {
			buf = append(buf, uint16(r))
		}
	}
	return buf
}
