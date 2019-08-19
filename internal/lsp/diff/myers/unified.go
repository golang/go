// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package myers

import (
	"fmt"
	"strings"
)

type Unified struct {
	From, To string
	Hunks    []*Hunk
}

type Hunk struct {
	FromLine int
	ToLine   int
	Lines    []Line
}

type Line struct {
	Kind    OpKind
	Content string
}

const (
	edge = 3
	gap  = edge * 2
)

func ToUnified(from, to string, lines []string, ops []*Op) Unified {
	u := Unified{
		From: from,
		To:   to,
	}
	if len(ops) == 0 {
		return u
	}
	var h *Hunk
	last := -(gap + 2)
	for _, op := range ops {
		switch {
		case op.I1 < last:
			panic("cannot convert unsorted operations to unified diff")
		case op.I1 == last:
			//direct extension
		case op.I1 <= last+gap:
			//within range of previous lines, add the joiners
			addEqualLines(h, lines, last, op.I1)
		default:
			//need to start a new hunk
			if h != nil {
				// add the edge to the previous hunk
				addEqualLines(h, lines, last, last+edge)
				u.Hunks = append(u.Hunks, h)
			}
			h = &Hunk{
				FromLine: op.I1 + 1,
				ToLine:   op.J1 + 1,
			}
			// add the edge to the new hunk
			delta := addEqualLines(h, lines, op.I1-edge, op.I1)
			h.FromLine -= delta
			h.ToLine -= delta
		}
		last = op.I1
		switch op.Kind {
		case Delete:
			for i := op.I1; i < op.I2; i++ {
				h.Lines = append(h.Lines, Line{Kind: Delete, Content: lines[i]})
				last++
			}
		case Insert:
			for _, c := range op.Content {
				h.Lines = append(h.Lines, Line{Kind: Insert, Content: c})
			}
		default:
			// all other op types ignored
		}
	}
	if h != nil {
		// add the edge to the final hunk
		addEqualLines(h, lines, last, last+edge)
		u.Hunks = append(u.Hunks, h)
	}
	return u
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

func (u Unified) Format(f fmt.State, r rune) {
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
