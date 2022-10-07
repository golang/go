// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package diff

import (
	"fmt"
	"log"
	"strings"
)

// Unified returns a unified diff of the old and new strings.
// The old and new labels are the names of the old and new files.
// If the strings are equal, it returns the empty string.
func Unified(oldLabel, newLabel, old, new string) string {
	edits := Strings(old, new)
	unified, err := ToUnified(oldLabel, newLabel, old, edits)
	if err != nil {
		// Can't happen: edits are consistent.
		log.Fatalf("internal error in diff.Unified: %v", err)
	}
	return unified
}

// ToUnified applies the edits to content and returns a unified diff.
// The old and new labels are the names of the content and result files.
// It returns an error if the edits are inconsistent; see ApplyEdits.
func ToUnified(oldLabel, newLabel, content string, edits []Edit) (string, error) {
	u, err := toUnified(oldLabel, newLabel, content, edits)
	if err != nil {
		return "", err
	}
	return u.String(), nil
}

// unified represents a set of edits as a unified diff.
type unified struct {
	// From is the name of the original file.
	From string
	// To is the name of the modified file.
	To string
	// Hunks is the set of edit hunks needed to transform the file content.
	Hunks []*hunk
}

// Hunk represents a contiguous set of line edits to apply.
type hunk struct {
	// The line in the original source where the hunk starts.
	FromLine int
	// The line in the original source where the hunk finishes.
	ToLine int
	// The set of line based edits to apply.
	Lines []line
}

// Line represents a single line operation to apply as part of a Hunk.
type line struct {
	// Kind is the type of line this represents, deletion, insertion or copy.
	Kind OpKind
	// Content is the content of this line.
	// For deletion it is the line being removed, for all others it is the line
	// to put in the output.
	Content string
}

// OpKind is used to denote the type of operation a line represents.
// TODO(adonovan): hide this once the myers package no longer references it.
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

// toUnified takes a file contents and a sequence of edits, and calculates
// a unified diff that represents those edits.
func toUnified(fromName, toName string, content string, edits []Edit) (unified, error) {
	u := unified{
		From: fromName,
		To:   toName,
	}
	if len(edits) == 0 {
		return u, nil
	}
	var err error
	edits, err = lineEdits(content, edits) // expand to whole lines
	if err != nil {
		return u, err
	}
	lines := splitLines(content)
	var h *hunk
	last := 0
	toLine := 0
	for _, edit := range edits {
		// Compute the zero-based line numbers of the edit start and end.
		// TODO(adonovan): opt: compute incrementally, avoid O(n^2).
		start := strings.Count(content[:edit.Start], "\n")
		end := strings.Count(content[:edit.End], "\n")
		if edit.End == len(content) && len(content) > 0 && content[len(content)-1] != '\n' {
			end++ // EOF counts as an implicit newline
		}

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
			h = &hunk{
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
			h.Lines = append(h.Lines, line{Kind: Delete, Content: lines[i]})
			last++
		}
		if edit.New != "" {
			for _, content := range splitLines(edit.New) {
				h.Lines = append(h.Lines, line{Kind: Insert, Content: content})
				toLine++
			}
		}
	}
	if h != nil {
		// add the edge to the final hunk
		addEqualLines(h, lines, last, last+edge)
		u.Hunks = append(u.Hunks, h)
	}
	return u, nil
}

func splitLines(text string) []string {
	lines := strings.SplitAfter(text, "\n")
	if lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	return lines
}

func addEqualLines(h *hunk, lines []string, start, end int) int {
	delta := 0
	for i := start; i < end; i++ {
		if i < 0 {
			continue
		}
		if i >= len(lines) {
			return delta
		}
		h.Lines = append(h.Lines, line{Kind: Equal, Content: lines[i]})
		delta++
	}
	return delta
}

// String converts a unified diff to the standard textual form for that diff.
// The output of this function can be passed to tools like patch.
func (u unified) String() string {
	if len(u.Hunks) == 0 {
		return ""
	}
	b := new(strings.Builder)
	fmt.Fprintf(b, "--- %s\n", u.From)
	fmt.Fprintf(b, "+++ %s\n", u.To)
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
		fmt.Fprint(b, "@@")
		if fromCount > 1 {
			fmt.Fprintf(b, " -%d,%d", hunk.FromLine, fromCount)
		} else if hunk.FromLine == 1 && fromCount == 0 {
			// Match odd GNU diff -u behavior adding to empty file.
			fmt.Fprintf(b, " -0,0")
		} else {
			fmt.Fprintf(b, " -%d", hunk.FromLine)
		}
		if toCount > 1 {
			fmt.Fprintf(b, " +%d,%d", hunk.ToLine, toCount)
		} else {
			fmt.Fprintf(b, " +%d", hunk.ToLine)
		}
		fmt.Fprint(b, " @@\n")
		for _, l := range hunk.Lines {
			switch l.Kind {
			case Delete:
				fmt.Fprintf(b, "-%s", l.Content)
			case Insert:
				fmt.Fprintf(b, "+%s", l.Content)
			default:
				fmt.Fprintf(b, " %s", l.Content)
			}
			if !strings.HasSuffix(l.Content, "\n") {
				fmt.Fprintf(b, "\n\\ No newline at end of file\n")
			}
		}
	}
	return b.String()
}
