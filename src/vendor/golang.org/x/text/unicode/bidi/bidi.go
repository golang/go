// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate go run gen.go gen_trieval.go gen_ranges.go

// Package bidi contains functionality for bidirectional text support.
//
// See https://www.unicode.org/reports/tr9.
//
// NOTE: UNDER CONSTRUCTION. This API may change in backwards incompatible ways
// and without notice.
package bidi // import "golang.org/x/text/unicode/bidi"

// TODO
// - Transformer for reordering?
// - Transformer (validator, really) for Bidi Rule.

import (
	"bytes"
)

// This API tries to avoid dealing with embedding levels for now. Under the hood
// these will be computed, but the question is to which extent the user should
// know they exist. We should at some point allow the user to specify an
// embedding hierarchy, though.

// A Direction indicates the overall flow of text.
type Direction int

const (
	// LeftToRight indicates the text contains no right-to-left characters and
	// that either there are some left-to-right characters or the option
	// DefaultDirection(LeftToRight) was passed.
	LeftToRight Direction = iota

	// RightToLeft indicates the text contains no left-to-right characters and
	// that either there are some right-to-left characters or the option
	// DefaultDirection(RightToLeft) was passed.
	RightToLeft

	// Mixed indicates text contains both left-to-right and right-to-left
	// characters.
	Mixed

	// Neutral means that text contains no left-to-right and right-to-left
	// characters and that no default direction has been set.
	Neutral
)

type options struct {
	defaultDirection Direction
}

// An Option is an option for Bidi processing.
type Option func(*options)

// ICU allows the user to define embedding levels. This may be used, for example,
// to use hierarchical structure of markup languages to define embeddings.
// The following option may be a way to expose this functionality in this API.
// // LevelFunc sets a function that associates nesting levels with the given text.
// // The levels function will be called with monotonically increasing values for p.
// func LevelFunc(levels func(p int) int) Option {
// 	panic("unimplemented")
// }

// DefaultDirection sets the default direction for a Paragraph. The direction is
// overridden if the text contains directional characters.
func DefaultDirection(d Direction) Option {
	return func(opts *options) {
		opts.defaultDirection = d
	}
}

// A Paragraph holds a single Paragraph for Bidi processing.
type Paragraph struct {
	p          []byte
	o          Ordering
	opts       []Option
	types      []Class
	pairTypes  []bracketType
	pairValues []rune
	runes      []rune
	options    options
}

// Initialize the p.pairTypes, p.pairValues and p.types from the input previously
// set by p.SetBytes() or p.SetString(). Also limit the input up to (and including) a paragraph
// separator (bidi class B).
//
// The function p.Order() needs these values to be set, so this preparation could be postponed.
// But since the SetBytes and SetStrings functions return the length of the input up to the paragraph
// separator, the whole input needs to be processed anyway and should not be done twice.
//
// The function has the same return values as SetBytes() / SetString()
func (p *Paragraph) prepareInput() (n int, err error) {
	p.runes = bytes.Runes(p.p)
	bytecount := 0
	// clear slices from previous SetString or SetBytes
	p.pairTypes = nil
	p.pairValues = nil
	p.types = nil

	for _, r := range p.runes {
		props, i := LookupRune(r)
		bytecount += i
		cls := props.Class()
		if cls == B {
			return bytecount, nil
		}
		p.types = append(p.types, cls)
		if props.IsOpeningBracket() {
			p.pairTypes = append(p.pairTypes, bpOpen)
			p.pairValues = append(p.pairValues, r)
		} else if props.IsBracket() {
			// this must be a closing bracket,
			// since IsOpeningBracket is not true
			p.pairTypes = append(p.pairTypes, bpClose)
			p.pairValues = append(p.pairValues, r)
		} else {
			p.pairTypes = append(p.pairTypes, bpNone)
			p.pairValues = append(p.pairValues, 0)
		}
	}
	return bytecount, nil
}

// SetBytes configures p for the given paragraph text. It replaces text
// previously set by SetBytes or SetString. If b contains a paragraph separator
// it will only process the first paragraph and report the number of bytes
// consumed from b including this separator. Error may be non-nil if options are
// given.
func (p *Paragraph) SetBytes(b []byte, opts ...Option) (n int, err error) {
	p.p = b
	p.opts = opts
	return p.prepareInput()
}

// SetString configures s for the given paragraph text. It replaces text
// previously set by SetBytes or SetString. If s contains a paragraph separator
// it will only process the first paragraph and report the number of bytes
// consumed from s including this separator. Error may be non-nil if options are
// given.
func (p *Paragraph) SetString(s string, opts ...Option) (n int, err error) {
	p.p = []byte(s)
	p.opts = opts
	return p.prepareInput()
}

// IsLeftToRight reports whether the principle direction of rendering for this
// paragraphs is left-to-right. If this returns false, the principle direction
// of rendering is right-to-left.
func (p *Paragraph) IsLeftToRight() bool {
	return p.Direction() == LeftToRight
}

// Direction returns the direction of the text of this paragraph.
//
// The direction may be LeftToRight, RightToLeft, Mixed, or Neutral.
func (p *Paragraph) Direction() Direction {
	return p.o.Direction()
}

// TODO: what happens if the position is > len(input)? This should return an error.

// RunAt reports the Run at the given position of the input text.
//
// This method can be used for computing line breaks on paragraphs.
func (p *Paragraph) RunAt(pos int) Run {
	c := 0
	runNumber := 0
	for i, r := range p.o.runes {
		c += len(r)
		if pos < c {
			runNumber = i
		}
	}
	return p.o.Run(runNumber)
}

func calculateOrdering(levels []level, runes []rune) Ordering {
	var curDir Direction

	prevDir := Neutral
	prevI := 0

	o := Ordering{}
	// lvl = 0,2,4,...: left to right
	// lvl = 1,3,5,...: right to left
	for i, lvl := range levels {
		if lvl%2 == 0 {
			curDir = LeftToRight
		} else {
			curDir = RightToLeft
		}
		if curDir != prevDir {
			if i > 0 {
				o.runes = append(o.runes, runes[prevI:i])
				o.directions = append(o.directions, prevDir)
				o.startpos = append(o.startpos, prevI)
			}
			prevI = i
			prevDir = curDir
		}
	}
	o.runes = append(o.runes, runes[prevI:])
	o.directions = append(o.directions, prevDir)
	o.startpos = append(o.startpos, prevI)
	return o
}

// Order computes the visual ordering of all the runs in a Paragraph.
func (p *Paragraph) Order() (Ordering, error) {
	if len(p.types) == 0 {
		return Ordering{}, nil
	}

	for _, fn := range p.opts {
		fn(&p.options)
	}
	lvl := level(-1)
	if p.options.defaultDirection == RightToLeft {
		lvl = 1
	}
	para, err := newParagraph(p.types, p.pairTypes, p.pairValues, lvl)
	if err != nil {
		return Ordering{}, err
	}

	levels := para.getLevels([]int{len(p.types)})

	p.o = calculateOrdering(levels, p.runes)
	return p.o, nil
}

// Line computes the visual ordering of runs for a single line starting and
// ending at the given positions in the original text.
func (p *Paragraph) Line(start, end int) (Ordering, error) {
	lineTypes := p.types[start:end]
	para, err := newParagraph(lineTypes, p.pairTypes[start:end], p.pairValues[start:end], -1)
	if err != nil {
		return Ordering{}, err
	}
	levels := para.getLevels([]int{len(lineTypes)})
	o := calculateOrdering(levels, p.runes[start:end])
	return o, nil
}

// An Ordering holds the computed visual order of runs of a Paragraph. Calling
// SetBytes or SetString on the originating Paragraph invalidates an Ordering.
// The methods of an Ordering should only be called by one goroutine at a time.
type Ordering struct {
	runes      [][]rune
	directions []Direction
	startpos   []int
}

// Direction reports the directionality of the runs.
//
// The direction may be LeftToRight, RightToLeft, Mixed, or Neutral.
func (o *Ordering) Direction() Direction {
	return o.directions[0]
}

// NumRuns returns the number of runs.
func (o *Ordering) NumRuns() int {
	return len(o.runes)
}

// Run returns the ith run within the ordering.
func (o *Ordering) Run(i int) Run {
	r := Run{
		runes:     o.runes[i],
		direction: o.directions[i],
		startpos:  o.startpos[i],
	}
	return r
}

// TODO: perhaps with options.
// // Reorder creates a reader that reads the runes in visual order per character.
// // Modifiers remain after the runes they modify.
// func (l *Runs) Reorder() io.Reader {
// 	panic("unimplemented")
// }

// A Run is a continuous sequence of characters of a single direction.
type Run struct {
	runes     []rune
	direction Direction
	startpos  int
}

// String returns the text of the run in its original order.
func (r *Run) String() string {
	return string(r.runes)
}

// Bytes returns the text of the run in its original order.
func (r *Run) Bytes() []byte {
	return []byte(r.String())
}

// TODO: methods for
// - Display order
// - headers and footers
// - bracket replacement.

// Direction reports the direction of the run.
func (r *Run) Direction() Direction {
	return r.direction
}

// Pos returns the position of the Run within the text passed to SetBytes or SetString of the
// originating Paragraph value.
func (r *Run) Pos() (start, end int) {
	return r.startpos, r.startpos + len(r.runes) - 1
}

// AppendReverse reverses the order of characters of in, appends them to out,
// and returns the result. Modifiers will still follow the runes they modify.
// Brackets are replaced with their counterparts.
func AppendReverse(out, in []byte) []byte {
	ret := make([]byte, len(in)+len(out))
	copy(ret, out)
	inRunes := bytes.Runes(in)

	for i, r := range inRunes {
		prop, _ := LookupRune(r)
		if prop.IsBracket() {
			inRunes[i] = prop.reverseBracket(r)
		}
	}

	for i, j := 0, len(inRunes)-1; i < j; i, j = i+1, j-1 {
		inRunes[i], inRunes[j] = inRunes[j], inRunes[i]
	}
	copy(ret[len(out):], string(inRunes))

	return ret
}

// ReverseString reverses the order of characters in s and returns a new string.
// Modifiers will still follow the runes they modify. Brackets are replaced with
// their counterparts.
func ReverseString(s string) string {
	input := []rune(s)
	li := len(input)
	ret := make([]rune, li)
	for i, r := range input {
		prop, _ := LookupRune(r)
		if prop.IsBracket() {
			ret[li-i-1] = prop.reverseBracket(r)
		} else {
			ret[li-i-1] = r
		}
	}
	return string(ret)
}
