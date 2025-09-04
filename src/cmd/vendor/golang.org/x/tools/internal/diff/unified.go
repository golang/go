// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package diff

import (
	"fmt"
	"log"
	"strings"
)

// DefaultContextLines is the number of unchanged lines of surrounding
// context displayed by Unified. Use ToUnified to specify a different value.
const DefaultContextLines = 3

// Unified returns a unified diff of the old and new strings.
// The old and new labels are the names of the old and new files.
// If the strings are equal, it returns the empty string.
func Unified(oldLabel, newLabel, old, new string) string {
	edits := Strings(old, new)
	unified, err := ToUnified(oldLabel, newLabel, old, edits, DefaultContextLines)
	if err != nil {
		// Can't happen: edits are consistent.
		log.Fatalf("internal error in diff.Unified: %v", err)
	}
	return unified
}

// ToUnified applies the edits to content and returns a unified diff,
// with contextLines lines of (unchanged) context around each diff hunk.
// The old and new labels are the names of the content and result files.
// It returns an error if the edits are inconsistent; see ApplyEdits.
func ToUnified(oldLabel, newLabel, content string, edits []Edit, contextLines int) (string, error) {
	u, err := toUnified(oldLabel, newLabel, content, edits, contextLines)
	if err != nil {
		return "", err
	}
	return u.String(), nil
}

// unified represents a set of edits as a unified diff.
type unified struct {
	// from is the name of the original file.
	from string
	// to is the name of the modified file.
	to string
	// hunks is the set of edit hunks needed to transform the file content.
	hunks []*hunk
}

// Hunk represents a contiguous set of line edits to apply.
type hunk struct {
	// The line in the original source where the hunk starts.
	fromLine int
	// The line in the original source where the hunk finishes.
	toLine int
	// The set of line based edits to apply.
	lines []line
}

// Line represents a single line operation to apply as part of a Hunk.
type line struct {
	// kind is the type of line this represents, deletion, insertion or copy.
	kind opKind
	// content is the content of this line.
	// For deletion it is the line being removed, for all others it is the line
	// to put in the output.
	content string
}

// opKind is used to denote the type of operation a line represents.
type opKind int

const (
	// opDelete is the operation kind for a line that is present in the input
	// but not in the output.
	opDelete opKind = iota
	// opInsert is the operation kind for a line that is new in the output.
	opInsert
	// opEqual is the operation kind for a line that is the same in the input and
	// output, often used to provide context around edited lines.
	opEqual
)

// String returns a human readable representation of an OpKind. It is not
// intended for machine processing.
func (k opKind) String() string {
	switch k {
	case opDelete:
		return "delete"
	case opInsert:
		return "insert"
	case opEqual:
		return "equal"
	default:
		panic("unknown operation kind")
	}
}

// toUnified takes a file contents and a sequence of edits, and calculates
// a unified diff that represents those edits.
func toUnified(fromName, toName string, content string, edits []Edit, contextLines int) (unified, error) {
	gap := contextLines * 2
	u := unified{
		from: fromName,
		to:   toName,
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
				addEqualLines(h, lines, last, last+contextLines)
				u.hunks = append(u.hunks, h)
			}
			toLine += start - last
			h = &hunk{
				fromLine: start + 1,
				toLine:   toLine + 1,
			}
			// add the edge to the new hunk
			delta := addEqualLines(h, lines, start-contextLines, start)
			h.fromLine -= delta
			h.toLine -= delta
		}
		last = start
		for i := start; i < end; i++ {
			h.lines = append(h.lines, line{kind: opDelete, content: lines[i]})
			last++
		}
		if edit.New != "" {
			for _, content := range splitLines(edit.New) {
				h.lines = append(h.lines, line{kind: opInsert, content: content})
				toLine++
			}
		}
	}
	if h != nil {
		// add the edge to the final hunk
		addEqualLines(h, lines, last, last+contextLines)
		u.hunks = append(u.hunks, h)
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
		h.lines = append(h.lines, line{kind: opEqual, content: lines[i]})
		delta++
	}
	return delta
}

// String converts a unified diff to the standard textual form for that diff.
// The output of this function can be passed to tools like patch.
func (u unified) String() string {
	if len(u.hunks) == 0 {
		return ""
	}
	b := new(strings.Builder)
	fmt.Fprintf(b, "--- %s\n", u.from)
	fmt.Fprintf(b, "+++ %s\n", u.to)
	for _, hunk := range u.hunks {
		fromCount, toCount := 0, 0
		for _, l := range hunk.lines {
			switch l.kind {
			case opDelete:
				fromCount++
			case opInsert:
				toCount++
			default:
				fromCount++
				toCount++
			}
		}
		fmt.Fprint(b, "@@")
		if fromCount > 1 {
			fmt.Fprintf(b, " -%d,%d", hunk.fromLine, fromCount)
		} else if hunk.fromLine == 1 && fromCount == 0 {
			// Match odd GNU diff -u behavior adding to empty file.
			fmt.Fprintf(b, " -0,0")
		} else {
			fmt.Fprintf(b, " -%d", hunk.fromLine)
		}
		if toCount > 1 {
			fmt.Fprintf(b, " +%d,%d", hunk.toLine, toCount)
		} else if hunk.toLine == 1 && toCount == 0 {
			// Match odd GNU diff -u behavior adding to empty file.
			fmt.Fprintf(b, " +0,0")
		} else {
			fmt.Fprintf(b, " +%d", hunk.toLine)
		}
		fmt.Fprint(b, " @@\n")
		for _, l := range hunk.lines {
			switch l.kind {
			case opDelete:
				fmt.Fprintf(b, "-%s", l.content)
			case opInsert:
				fmt.Fprintf(b, "+%s", l.content)
			default:
				fmt.Fprintf(b, " %s", l.content)
			}
			if !strings.HasSuffix(l.content, "\n") {
				fmt.Fprintf(b, "\n\\ No newline at end of file\n")
			}
		}
	}
	return b.String()
}
