// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package comment

import (
	"bytes"
	"fmt"
	"sort"
	"strings"
	"unicode/utf8"
)

// A textPrinter holds the state needed for printing a Doc as plain text.
type textPrinter struct {
	*Printer
	long       strings.Builder
	prefix     string
	codePrefix string
	width      int
}

// Text returns a textual formatting of the [Doc].
// See the [Printer] documentation for ways to customize the text output.
func (p *Printer) Text(d *Doc) []byte {
	tp := &textPrinter{
		Printer:    p,
		prefix:     p.TextPrefix,
		codePrefix: p.TextCodePrefix,
		width:      p.TextWidth,
	}
	if tp.codePrefix == "" {
		tp.codePrefix = p.TextPrefix + "\t"
	}
	if tp.width == 0 {
		tp.width = 80 - utf8.RuneCountInString(tp.prefix)
	}

	var out bytes.Buffer
	for i, x := range d.Content {
		if i > 0 && blankBefore(x) {
			out.WriteString(tp.prefix)
			writeNL(&out)
		}
		tp.block(&out, x)
	}
	anyUsed := false
	for _, def := range d.Links {
		if def.Used {
			anyUsed = true
			break
		}
	}
	if anyUsed {
		writeNL(&out)
		for _, def := range d.Links {
			if def.Used {
				fmt.Fprintf(&out, "[%s]: %s\n", def.Text, def.URL)
			}
		}
	}
	return out.Bytes()
}

// writeNL calls out.WriteByte('\n')
// but first trims trailing spaces on the previous line.
func writeNL(out *bytes.Buffer) {
	// Trim trailing spaces.
	data := out.Bytes()
	n := 0
	for n < len(data) && (data[len(data)-n-1] == ' ' || data[len(data)-n-1] == '\t') {
		n++
	}
	if n > 0 {
		out.Truncate(len(data) - n)
	}
	out.WriteByte('\n')
}

// block prints the block x to out.
func (p *textPrinter) block(out *bytes.Buffer, x Block) {
	switch x := x.(type) {
	default:
		fmt.Fprintf(out, "?%T\n", x)

	case *Paragraph:
		out.WriteString(p.prefix)
		p.text(out, "", x.Text)

	case *Heading:
		out.WriteString(p.prefix)
		out.WriteString("# ")
		p.text(out, "", x.Text)

	case *Code:
		text := x.Text
		for text != "" {
			var line string
			line, text, _ = strings.Cut(text, "\n")
			if line != "" {
				out.WriteString(p.codePrefix)
				out.WriteString(line)
			}
			writeNL(out)
		}

	case *List:
		loose := x.BlankBetween()
		for i, item := range x.Items {
			if i > 0 && loose {
				out.WriteString(p.prefix)
				writeNL(out)
			}
			out.WriteString(p.prefix)
			out.WriteString(" ")
			if item.Number == "" {
				out.WriteString(" - ")
			} else {
				out.WriteString(item.Number)
				out.WriteString(". ")
			}
			for i, blk := range item.Content {
				const fourSpace = "    "
				if i > 0 {
					writeNL(out)
					out.WriteString(p.prefix)
					out.WriteString(fourSpace)
				}
				p.text(out, fourSpace, blk.(*Paragraph).Text)
			}
		}
	}
}

// text prints the text sequence x to out.
func (p *textPrinter) text(out *bytes.Buffer, indent string, x []Text) {
	p.oneLongLine(&p.long, x)
	words := strings.Fields(p.long.String())
	p.long.Reset()

	var seq []int
	if p.width < 0 || len(words) == 0 {
		seq = []int{0, len(words)} // one long line
	} else {
		seq = wrap(words, p.width-utf8.RuneCountInString(indent))
	}
	for i := 0; i+1 < len(seq); i++ {
		if i > 0 {
			out.WriteString(p.prefix)
			out.WriteString(indent)
		}
		for j, w := range words[seq[i]:seq[i+1]] {
			if j > 0 {
				out.WriteString(" ")
			}
			out.WriteString(w)
		}
		writeNL(out)
	}
}

// oneLongLine prints the text sequence x to out as one long line,
// without worrying about line wrapping.
// Explicit links have the [ ] dropped to improve readability.
func (p *textPrinter) oneLongLine(out *strings.Builder, x []Text) {
	for _, t := range x {
		switch t := t.(type) {
		case Plain:
			out.WriteString(string(t))
		case Italic:
			out.WriteString(string(t))
		case *Link:
			p.oneLongLine(out, t.Text)
		case *DocLink:
			p.oneLongLine(out, t.Text)
		}
	}
}

// wrap wraps words into lines of at most max runes,
// minimizing the sum of the squares of the leftover lengths
// at the end of each line (except the last, of course),
// with a preference for ending lines at punctuation (.,:;).
//
// The returned slice gives the indexes of the first words
// on each line in the wrapped text with a final entry of len(words).
// Thus the lines are words[seq[0]:seq[1]], words[seq[1]:seq[2]],
// ..., words[seq[len(seq)-2]:seq[len(seq)-1]].
//
// The implementation runs in O(n log n) time, where n = len(words),
// using the algorithm described in D. S. Hirschberg and L. L. Larmore,
// “[The least weight subsequence problem],” FOCS 1985, pp. 137-143.
//
// [The least weight subsequence problem]: https://doi.org/10.1109/SFCS.1985.60
func wrap(words []string, max int) (seq []int) {
	// The algorithm requires that our scoring function be concave,
	// meaning that for all i₀ ≤ i₁ < j₀ ≤ j₁,
	// weight(i₀, j₀) + weight(i₁, j₁) ≤ weight(i₀, j₁) + weight(i₁, j₀).
	//
	// Our weights are two-element pairs [hi, lo]
	// ordered by elementwise comparison.
	// The hi entry counts the weight for lines that are longer than max,
	// and the lo entry counts the weight for lines that are not.
	// This forces the algorithm to first minimize the number of lines
	// that are longer than max, which correspond to lines with
	// single very long words. Having done that, it can move on to
	// minimizing the lo score, which is more interesting.
	//
	// The lo score is the sum for each line of the square of the
	// number of spaces remaining at the end of the line and a
	// penalty of 64 given out for not ending the line in a
	// punctuation character (.,:;).
	// The penalty is somewhat arbitrarily chosen by trying
	// different amounts and judging how nice the wrapped text looks.
	// Roughly speaking, using 64 means that we are willing to
	// end a line with eight blank spaces in order to end at a
	// punctuation character, even if the next word would fit in
	// those spaces.
	//
	// We care about ending in punctuation characters because
	// it makes the text easier to skim if not too many sentences
	// or phrases begin with a single word on the previous line.

	// A score is the score (also called weight) for a given line.
	// add and cmp add and compare scores.
	type score struct {
		hi int64
		lo int64
	}
	add := func(s, t score) score { return score{s.hi + t.hi, s.lo + t.lo} }
	cmp := func(s, t score) int {
		switch {
		case s.hi < t.hi:
			return -1
		case s.hi > t.hi:
			return +1
		case s.lo < t.lo:
			return -1
		case s.lo > t.lo:
			return +1
		}
		return 0
	}

	// total[j] is the total number of runes
	// (including separating spaces) in words[:j].
	total := make([]int, len(words)+1)
	total[0] = 0
	for i, s := range words {
		total[1+i] = total[i] + utf8.RuneCountInString(s) + 1
	}

	// weight returns weight(i, j).
	weight := func(i, j int) score {
		// On the last line, there is zero weight for being too short.
		n := total[j] - 1 - total[i]
		if j == len(words) && n <= max {
			return score{0, 0}
		}

		// Otherwise the weight is the penalty plus the square of the number of
		// characters remaining on the line or by which the line goes over.
		// In the latter case, that value goes in the hi part of the score.
		// (See note above.)
		p := wrapPenalty(words[j-1])
		v := int64(max-n) * int64(max-n)
		if n > max {
			return score{v, p}
		}
		return score{0, v + p}
	}

	// The rest of this function is “The Basic Algorithm” from
	// Hirschberg and Larmore's conference paper,
	// using the same names as in the paper.
	f := []score{{0, 0}}
	g := func(i, j int) score { return add(f[i], weight(i, j)) }

	bridge := func(a, b, c int) bool {
		k := c + sort.Search(len(words)+1-c, func(k int) bool {
			k += c
			return cmp(g(a, k), g(b, k)) > 0
		})
		if k > len(words) {
			return true
		}
		return cmp(g(c, k), g(b, k)) <= 0
	}

	// d is a one-ended deque implemented as a slice.
	d := make([]int, 1, len(words))
	d[0] = 0
	bestleft := make([]int, 1, len(words))
	bestleft[0] = -1
	for m := 1; m < len(words); m++ {
		f = append(f, g(d[0], m))
		bestleft = append(bestleft, d[0])
		for len(d) > 1 && cmp(g(d[1], m+1), g(d[0], m+1)) <= 0 {
			d = d[1:] // “Retire”
		}
		for len(d) > 1 && bridge(d[len(d)-2], d[len(d)-1], m) {
			d = d[:len(d)-1] // “Fire”
		}
		if cmp(g(m, len(words)), g(d[len(d)-1], len(words))) < 0 {
			d = append(d, m) // “Hire”
			// The next few lines are not in the paper but are necessary
			// to handle two-word inputs correctly. It appears to be
			// just a bug in the paper's pseudocode.
			if len(d) == 2 && cmp(g(d[1], m+1), g(d[0], m+1)) <= 0 {
				d = d[1:]
			}
		}
	}
	bestleft = append(bestleft, d[0])

	// Recover least weight sequence from bestleft.
	n := 1
	for m := len(words); m > 0; m = bestleft[m] {
		n++
	}
	seq = make([]int, n)
	for m := len(words); m > 0; m = bestleft[m] {
		n--
		seq[n] = m
	}
	return seq
}

// wrapPenalty is the penalty for inserting a line break after word s.
func wrapPenalty(s string) int64 {
	switch s[len(s)-1] {
	case '.', ',', ':', ';':
		return 0
	}
	return 64
}
