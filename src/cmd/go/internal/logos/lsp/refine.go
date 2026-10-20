// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import "bytes"

// maxRefineCells bounds the size of the token table of one replaced span.
// A larger span keeps its coarse edit: it maps by span, not by token.
const maxRefineCells = 1 << 20

// token is the byte range of one lexical unit of a text.
type token struct{ start, end int }

// run is a stretch of text that appears unchanged in both the original
// and the lowered text of a replaced span, as byte offsets relative to the
// start of each text.
type run struct{ o0, o1, l0, l1 int }

// charClass is the lexical class of a byte for refinement.
type charClass int

const (
	classWord charClass = iota
	classSpace
	classPunct
)

// classOf returns the class of c. Bytes of multi-byte characters count as
// word bytes, so an identifier with such characters stays one token.
func classOf(c byte) charClass {
	switch {
	case c == '_' || c >= 0x80 || c >= '0' && c <= '9' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z':
		return classWord
	case c == ' ' || c == '\t' || c == '\n' || c == '\r':
		return classSpace
	}
	return classPunct
}

// refineSpans splits every edit that replaces text with text into the
// smaller edits left once the stretches the lowering copied unchanged are
// taken out. The lowering of a ? statement, for one, replaces the whole
// statement with a rewritten copy of it; refinement keeps the operand
// mapped exactly and leaves only the generated pieces as edits.
func refineSpans(src, out []byte, edits []EditSpan) []EditSpan {
	refined := make([]EditSpan, 0, len(edits))
	shift := 0
	for _, s := range edits {
		start := s.Start + shift
		refined = append(refined, refineSpan(s, src[s.Start:s.End], out[start:start+s.Len])...)
		shift += s.Len - (s.End - s.Start)
	}
	return refined
}

// refineSpan splits one edit around the runs its texts share.
func refineSpan(s EditSpan, orig, low []byte) []EditSpan {
	var parts []EditSpan
	po, pl := 0, 0
	for _, r := range commonRuns(orig, low) {
		if r.o0 > po || r.l0 > pl {
			parts = append(parts, EditSpan{Start: s.Start + po, End: s.Start + r.o0, Len: r.l0 - pl})
		}
		po, pl = r.o1, r.l1
	}
	if po < len(orig) || pl < len(low) {
		parts = append(parts, EditSpan{Start: s.Start + po, End: s.End, Len: len(low) - pl})
	}
	return parts
}

// commonRuns returns the longest common token subsequence of orig and low
// as runs of consecutive tokens, left to right. A run needs a word token:
// a stretch of punctuation or space alone would anchor positions to a
// look-alike far from where they belong.
func commonRuns(orig, low []byte) []run {
	a, b := tokenize(orig), tokenize(low)
	if (len(a)+1)*(len(b)+1) > maxRefineCells {
		return nil
	}
	table := lcsTable(orig, a, low, b)
	var runs []run
	for _, pair := range lcsPairs(orig, a, low, b, table) {
		runs = extendRuns(runs, a[pair[0]], b[pair[1]])
	}
	return wordRuns(orig, runs)
}

// extendRuns appends the matched token pair (x, y) to runs, joining it to
// the last run when both tokens touch it.
func extendRuns(runs []run, x, y token) []run {
	if n := len(runs); n > 0 && runs[n-1].o1 == x.start && runs[n-1].l1 == y.start {
		runs[n-1].o1, runs[n-1].l1 = x.end, y.end
		return runs
	}
	return append(runs, run{x.start, x.end, y.start, y.end})
}

// wordRuns keeps the runs of orig that hold a word byte.
func wordRuns(orig []byte, runs []run) []run {
	var kept []run
	for _, r := range runs {
		if hasWord(orig[r.o0:r.o1]) {
			kept = append(kept, r)
		}
	}
	return kept
}

// hasWord reports whether b holds a word byte.
func hasWord(b []byte) bool {
	for _, c := range b {
		if classOf(c) == classWord {
			return true
		}
	}
	return false
}

// tokenize splits b into runs of word bytes, runs of space bytes, and
// single punctuation bytes.
func tokenize(b []byte) []token {
	var toks []token
	for i := 0; i < len(b); {
		j := i + 1
		if c := classOf(b[i]); c != classPunct {
			for j < len(b) && classOf(b[j]) == c {
				j++
			}
		}
		toks = append(toks, token{i, j})
		i = j
	}
	return toks
}

// lcsTable returns the dynamic programming table of the longest common
// subsequence of the token lists a and b: cell i*(len(b)+1)+j holds the
// length for the suffixes starting at a[i] and b[j].
func lcsTable(orig []byte, a []token, low []byte, b []token) []int32 {
	m := len(b) + 1
	table := make([]int32, (len(a)+1)*m)
	for i := len(a) - 1; i >= 0; i-- {
		for j := len(b) - 1; j >= 0; j-- {
			if sameToken(orig, a[i], low, b[j]) {
				table[i*m+j] = table[(i+1)*m+j+1] + 1
			} else {
				table[i*m+j] = max(table[(i+1)*m+j], table[i*m+j+1])
			}
		}
	}
	return table
}

// lcsPairs walks table from the start and returns the matched token index
// pairs of one longest common subsequence.
func lcsPairs(orig []byte, a []token, low []byte, b []token, table []int32) [][2]int {
	m := len(b) + 1
	var pairs [][2]int
	for i, j := 0, 0; i < len(a) && j < len(b); {
		switch {
		case sameToken(orig, a[i], low, b[j]) && table[i*m+j] == table[(i+1)*m+j+1]+1:
			pairs = append(pairs, [2]int{i, j})
			i, j = i+1, j+1
		case table[(i+1)*m+j] >= table[i*m+j+1]:
			i++
		default:
			j++
		}
	}
	return pairs
}

// sameToken reports whether token x of orig and token y of low spell the
// same bytes.
func sameToken(orig []byte, x token, low []byte, y token) bool {
	return bytes.Equal(orig[x.start:x.end], low[y.start:y.end])
}
