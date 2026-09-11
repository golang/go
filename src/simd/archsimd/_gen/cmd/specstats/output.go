// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"strings"
)

// labelWidth leaves one space after the longest label. Increase this if labels
// change and the tables get out of alignment.
const labelWidth = 47

// section writes a section heading and the tasks that move the figures in it.
// A tag long enough to push the heading past the line budget goes on its own
// line rather than trailing off the screen.
func section(w *bytes.Buffer, title, tasks string) {
	head := "## " + title
	if tag := taskTag(tasks); tag != "" {
		if len(head)+len(tag)+1 <= 88 {
			head += " " + tag
		} else {
			head += "\n   " + strings.TrimSpace(tag)
		}
	}
	fmt.Fprintf(w, "\n\n%s\n\n", head)
}

// note writes a block of explanatory prose, followed by a blank line.
func note(w *bytes.Buffer, text string) {
	fmt.Fprintf(w, "%s\n\n", text)
}

// frac writes a figure heading for 100%.
func frac(w *bytes.Buffer, label string, r result, tasks string) {
	pct := "  --"
	if r.d > 0 {
		pct = fmt.Sprintf("%3.0f%%", 100*float64(r.n)/float64(r.d))
	}
	line(w, fmt.Sprintf("%-*s %5d / %-5d %s%s", labelWidth, label, r.n, r.d, pct, taskTag(tasks)))
}

// count writes a figure heading for zero.
func count(w *bytes.Buffer, label string, n int, tasks string) {
	line(w, fmt.Sprintf("%-*s %5d %-9s%s", labelWidth, label, n, "-> 0", taskTag(tasks)))
}

// kv writes a bare number: context rather than a figure with a target.
func kv(w *bytes.Buffer, label string, n int) {
	line(w, fmt.Sprintf("%-*s %5d", labelWidth, label, n))
}

// sub writes a bare number indented under the figure it breaks down.
func sub(w *bytes.Buffer, label string, n int) {
	line(w, fmt.Sprintf("    %-*s %5d", labelWidth-4, label, n))
}

// countIndented writes a count heading for zero, indented under the figure it
// breaks down. Used for per-field and per-file detail rows.
func countIndented(w *bytes.Buffer, label string, n int, tasks string) {
	line(w, fmt.Sprintf("    %-*s %5d %-9s%s", labelWidth-4, label, n, "-> 0", taskTag(tasks)))
}

// emptyList writes a placeholder when a list has no rows, so that a heading
// with nothing under it reads as an empty list rather than a truncated one.
func emptyList(w *bytes.Buffer, n int) {
	if n == 0 {
		line(w, "    (none)")
	}
}

const lineWrap = 78

// wrap writes lead followed by a list of words, breaking before the line
// wrapping budget and aligning continuations under the first word. The lead
// labels the list, so that an inventory nested under a table row says what it
// is rather than leaving the reader to infer it from position.
func wrap(w *bytes.Buffer, indent, lead string, words []string) {
	first, rest := indent+lead, indent+strings.Repeat(" ", len(lead))
	cur := first
	for _, s := range words {
		if len(cur) > len(rest) && len(cur)+1+len(s) > lineWrap {
			line(w, cur)
			cur = rest
		}
		if len(cur) > len(rest) {
			cur += " "
		}
		cur += s
	}
	if len(cur) > len(rest) {
		line(w, cur)
	}
}

func line(w *bytes.Buffer, s string) {
	fmt.Fprintf(w, "%s\n", strings.TrimRight(s, " "))
}

func taskTag(tasks string) string {
	if tasks == "" || tasks == "-" {
		return ""
	}
	return "  [" + tasks + "]"
}
