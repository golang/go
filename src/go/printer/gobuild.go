// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package printer

import (
	"go/build/constraint"
	"slices"
	"text/tabwriter"
)

func (p *printer) fixGoBuildLines() {
	if len(p.goBuild)+len(p.plusBuild) == 0 {
		return
	}

	// Find latest possible placement of //go:build and // +build comments.
	// That's just after the last blank line before we find a non-comment.
	// (We'll add another blank line after our comment block.)
	// When we start dropping // +build comments, we can skip over /* */ comments too.
	// Note that we are processing tabwriter input, so every comment
	// begins and ends with a tabwriter.Escape byte.
	// And some newlines have turned into \f bytes.
	insert := 0
	for pos := 0; ; {
		// Skip leading space at beginning of line.
		blank := true
		for pos < len(p.output) && (p.output[pos] == ' ' || p.output[pos] == '\t') {
			pos++
		}
		// Skip over // comment if any.
		if pos+3 < len(p.output) && p.output[pos] == tabwriter.Escape && p.output[pos+1] == '/' && p.output[pos+2] == '/' {
			blank = false
			for pos < len(p.output) && !isNL(p.output[pos]) {
				pos++
			}
		}
		// Skip over \n at end of line.
		if pos >= len(p.output) || !isNL(p.output[pos]) {
			break
		}
		pos++

		if blank {
			insert = pos
		}
	}

	// If there is a //go:build comment before the place we identified,
	// use that point instead. (Earlier in the file is always fine.)
	if len(p.goBuild) > 0 && p.goBuild[0] < insert {
		insert = p.goBuild[0]
	} else if len(p.plusBuild) > 0 && p.plusBuild[0] < insert {
		insert = p.plusBuild[0]
	}

	var x constraint.Expr
	switch len(p.goBuild) {
	case 0:
		// Synthesize //go:build expression from // +build lines.
		for _, pos := range p.plusBuild {
			y, err := constraint.Parse(p.commentTextAt(pos))
			if err != nil {
				x = nil
				break
			}
			if x == nil {
				x = y
			} else {
				x = &constraint.AndExpr{X: x, Y: y}
			}
		}
	case 1:
		// Parse //go:build expression.
		x, _ = constraint.Parse(p.commentTextAt(p.goBuild[0]))
	}

	var block []byte
	if x == nil {
		// Don't have a valid //go:build expression to treat as truth.
		// Bring all the lines together but leave them alone.
		// Note that these are already tabwriter-escaped.
		for _, pos := range p.goBuild {
			block = append(block, p.lineAt(pos)...)
		}
		for _, pos := range p.plusBuild {
			block = append(block, p.lineAt(pos)...)
		}
	} else {
		block = append(block, tabwriter.Escape)
		block = append(block, "//go:build "...)
		block = append(block, x.String()...)
		block = append(block, tabwriter.Escape, '\n')
		if len(p.plusBuild) > 0 {
			lines, err := constraint.PlusBuildLines(x)
			if err != nil {
				lines = []string{"// +build error: " + err.Error()}
			}
			for _, line := range lines {
				block = append(block, tabwriter.Escape)
				block = append(block, line...)
				block = append(block, tabwriter.Escape, '\n')
			}
		}
	}
	block = append(block, '\n')

	// Build sorted list of lines to delete from remainder of output.
	toDelete := append(p.goBuild, p.plusBuild...)
	slices.Sort(toDelete)

	// Collect output after insertion point, with lines deleted, into after.
	var after []byte
	start := insert
	for _, end := range toDelete {
		if end < start {
			continue
		}
		after = appendLines(after, p.output[start:end])
		start = end + len(p.lineAt(end))
	}
	after = appendLines(after, p.output[start:])
	if n := len(after); n >= 2 && isNL(after[n-1]) && isNL(after[n-2]) {
		after = after[:n-1]
	}

	p.output = p.output[:insert]
	p.output = append(p.output, block...)
	p.output = append(p.output, after...)
}

// appendLines is like append(x, y...)
// but it avoids creating doubled blank lines,
// which would not be gofmt-standard output.
// It assumes that only whole blocks of lines are being appended,
// not line fragments.
func appendLines(x, y []byte) []byte {
	if len(y) > 0 && isNL(y[0]) && // y starts in blank line
		(len(x) == 0 || len(x) >= 2 && isNL(x[len(x)-1]) && isNL(x[len(x)-2])) { // x is empty or ends in blank line
		y = y[1:] // delete y's leading blank line
	}
	return append(x, y...)
}

func (p *printer) lineAt(start int) []byte {
	pos := start
	for pos < len(p.output) && !isNL(p.output[pos]) {
		pos++
	}
	if pos < len(p.output) {
		pos++
	}
	return p.output[start:pos]
}

func (p *printer) commentTextAt(start int) string {
	if start < len(p.output) && p.output[start] == tabwriter.Escape {
		start++
	}
	pos := start
	for pos < len(p.output) && p.output[pos] != tabwriter.Escape && !isNL(p.output[pos]) {
		pos++
	}
	return string(p.output[start:pos])
}

func isNL(b byte) bool {
	return b == '\n' || b == '\f'
}
