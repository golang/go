// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package diff

import (
	"fmt"
	"strings"
)

// Unified represents a set of edits as a unified diff.
type Unified struct {
	// From is the name of the original file.
	From string
	// To is the name of the modified file.
	To string
	// Hunks is the set of edit hunks needed to transform the file content.
	Hunks []*Hunk
}

// Hunk represents a contiguous set of line edits to apply.
type Hunk struct {
	// The line in the original source where the hunk starts.
	FromLine int
	// The line in the original source where the hunk finishes.
	ToLine int
	// The set of line based edits to apply.
	Lines []Line
}

// Line represents a single line operation to apply as part of a Hunk.
type Line struct {
	// Kind is the type of line this represents, deletion, insertion or copy.
	Kind OpKind
	// Content is the content of this line.
	// For deletion it is the line being removed, for all others it is the line
	// to put in the output.
	Content string
}

// OpKind is used to denote the type of operation a line represents.
type OpKind int

const (
	// Delete is the operation kind for a line that is present in the input
	// but not in the output.
	Delete OpKind = iota
	// Insert is the operation kind for a line that is new in the output.
	Insert
	// Equal is the operation kind for a line that is the same in the input and
	// output, often used to provide context around edited lines.
	Equal
)

// String returns a human readable representation of an OpKind. It is not
// intended for machine processing.
func (k OpKind) String() string {
	switch k {
	case Delete:
		return "delete"
	case Insert:
		return "insert"
	case Equal:
		return "equal"
	default:
		panic("unknown operation kind")
	}
}

const (
	edge = 3
	gap  = edge * 2
)

// ToUnified takes a file contents and a sequence of edits, and calculates
// a unified diff that represents those edits.
func ToUnified(from, to string, content string, edits []TextEdit) Unified {
	u := Unified{
		From: from,
		To:   to,
	}
	if len(edits) == 0 {
		return u
	}
	edits, partial := prepareEdits(content, edits)
	if partial {
		edits = lineEdits(content, edits)
	}
	lines := splitLines(content)
	var h *Hunk
	last := 0
	toLine := 0
	for _, edit := range edits {
		start := edit.Span.Start().Line() - 1
		end := edit.Span.End().Line() - 1
		switch {
		case h != nil && start == last:
			//direct extension
		case h != nil && start <= last+gap:
			//within range of previous lines, add the joiners
			addEqualLines(h, lines, last, start)
		default:
			//need to start a new hunk
			if h != nil {
				// add the edge to the previous hunk
				addEqualLines(h, lines, last, last+edge)
				u.Hunks = append(u.Hunks, h)
			}
			toLine += start - last
			h = &Hunk{
				FromLine: start + 1,
				ToLine:   toLine + 1,
			}
			// add the edge to the new hunk
			delta := addEqualLines(h, lines, start-edge, start)
			h.FromLine -= delta
			h.ToLine -= delta
		}
		last = start
		for i := start; i < end; i++ {
			h.Lines = append(h.Lines, Line{Kind: Delete, Content: lines[i]})
			last++
		}
		if edit.NewText != "" {
			for _, line := range splitLines(edit.NewText) {
				h.Lines = append(h.Lines, Line{Kind: Insert, Content: line})
				toLine++
			}
		}
	}
	if h != nil {
		// add the edge to the final hunk
		addEqualLines(h, lines, last, last+edge)
		u.Hunks = append(u.Hunks, h)
	}
	return u
}

func splitLines(text string) []string {
	lines := strings.SplitAfter(text, "\n")
	if lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	return lines
}

func addEqualLines(h *Hunk, lines []string, start, end int) int {
	delta := 0
	for i := start; i < end; i++ {
		if i < 0 {
			continue
		}
		if i >= len(lines) {
			return delta
		}
		h.Lines = append(h.Lines, Line{Kind: Equal, Content: lines[i]})
		delta++
	}
	return delta
}

// Format converts a unified diff to the standard textual form for that diff.
// The output of this function can be passed to tools like patch.
func (u Unified) Format(f fmt.State, r rune) {
	if len(u.Hunks) == 0 {
		return
	}
	fmt.Fprintf(f, "--- %s\n", u.From)
	fmt.Fprintf(f, "+++ %s\n", u.To)
	for _, hunk := range u.Hunks {
		fromCount, toCount := 0, 0
		for _, l := range hunk.Lines {
			switch l.Kind {
			case Delete:
				fromCount++
			case Insert:
				toCount++
			default:
				fromCount++
				toCount++
			}
		}
		fmt.Fprint(f, "@@")
		if fromCount > 1 {
			fmt.Fprintf(f, " -%d,%d", hunk.FromLine, fromCount)
		} else {
			fmt.Fprintf(f, " -%d", hunk.FromLine)
		}
		if toCount > 1 {
			fmt.Fprintf(f, " +%d,%d", hunk.ToLine, toCount)
		} else {
			fmt.Fprintf(f, " +%d", hunk.ToLine)
		}
		fmt.Fprint(f, " @@\n")
		for _, l := range hunk.Lines {
			switch l.Kind {
			case Delete:
				fmt.Fprintf(f, "-%s", l.Content)
			case Insert:
				fmt.Fprintf(f, "+%s", l.Content)
			default:
				fmt.Fprintf(f, " %s", l.Content)
			}
			if !strings.HasSuffix(l.Content, "\n") {
				fmt.Fprintf(f, "\n\\ No newline at end of file\n")
			}
		}
	}
}
