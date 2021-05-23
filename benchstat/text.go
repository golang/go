// Copyright 2017 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package benchstat

import (
	"fmt"
	"io"
	"unicode/utf8"
)

// FormatText appends a fixed-width text formatting of the tables to w.
func FormatText(w io.Writer, tables []*Table) {
	var textTables [][]*textRow
	for _, t := range tables {
		textTables = append(textTables, toText(t))
	}

	var max []int
	for _, table := range textTables {
		for _, row := range table {
			if len(row.cols) == 1 {
				// Header row
				continue
			}
			for len(max) < len(row.cols) {
				max = append(max, 0)
			}
			for i, s := range row.cols {
				n := utf8.RuneCountInString(s)
				if max[i] < n {
					max[i] = n
				}
			}
		}
	}

	for i, table := range textTables {
		if i > 0 {
			fmt.Fprintf(w, "\n")
		}

		// headings
		row := table[0]
		for i, s := range row.cols {
			switch i {
			case 0:
				fmt.Fprintf(w, "%-*s", max[i], s)
			default:
				fmt.Fprintf(w, "  %-*s", max[i], s)
			case len(row.cols) - 1:
				fmt.Fprintf(w, "  %s\n", s)
			}
		}

		// data
		for _, row := range table[1:] {
			for i, s := range row.cols {
				switch {
				case len(row.cols) == 1:
					// Header row
					fmt.Fprint(w, s)
				case i == 0:
					fmt.Fprintf(w, "%-*s", max[i], s)
				default:
					if i == len(row.cols)-1 && len(s) > 0 && s[0] == '(' {
						// Left-align p value.
						fmt.Fprintf(w, "  %s", s)
						break
					}
					fmt.Fprintf(w, "  %*s", max[i], s)
				}
			}
			fmt.Fprintf(w, "\n")
		}
	}
}

// A textRow is a row of printed text columns.
type textRow struct {
	cols []string
}

func newTextRow(cols ...string) *textRow {
	return &textRow{cols: cols}
}

func (r *textRow) add(col string) {
	r.cols = append(r.cols, col)
}

func (r *textRow) trim() {
	for len(r.cols) > 0 && r.cols[len(r.cols)-1] == "" {
		r.cols = r.cols[:len(r.cols)-1]
	}
}

// toText converts the Table to a textual grid of cells,
// which can then be printed in fixed-width output.
func toText(t *Table) []*textRow {
	var textRows []*textRow
	switch len(t.Configs) {
	case 1:
		textRows = append(textRows, newTextRow("name", t.Metric))
	case 2:
		textRows = append(textRows, newTextRow("name", "old "+t.Metric, "new "+t.Metric, "delta"))
	default:
		row := newTextRow("name \\ " + t.Metric)
		row.cols = append(row.cols, t.Configs...)
		textRows = append(textRows, row)
	}

	var group string

	for _, row := range t.Rows {
		if row.Group != group {
			group = row.Group
			textRows = append(textRows, newTextRow(group))
		}
		text := newTextRow(row.Benchmark)
		for _, m := range row.Metrics {
			text.cols = append(text.cols, m.Format(row.Scaler))
		}
		if len(t.Configs) == 2 {
			delta := row.Delta
			if delta == "~" {
				delta = "~   "
			}
			text.cols = append(text.cols, delta)
			text.cols = append(text.cols, row.Note)
		}
		textRows = append(textRows, text)
	}
	for _, r := range textRows {
		r.trim()
	}
	return textRows
}
// Copyright 2017 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package benchstat

import (
	"encoding/csv"
	"fmt"
	"io"
	"path/filepath"
	"strings"
	"unicode/utf8"
)

// FormatText appends a fixed-width text formatting of the tables to w.
func FormatText(w io.Writer, tables []*Table) {
	var textTables [][]*textRow
	for _, t := range tables {
		textTables = append(textTables, toText(t))
	}

	var max []int
	for _, table := range textTables {
		for _, row := range table {
			if len(row.cols) == 1 {
				// Header row
				continue
			}
			for len(max) < len(row.cols) {
				max = append(max, 0)
			}
			for i, s := range row.cols {
				n := utf8.RuneCountInString(s)
				if max[i] < n {
					max[i] = n
				}
			}
		}
	}

	for i, table := range textTables {
		if i > 0 {
			fmt.Fprintf(w, "\n")
		}

		// headings
		row := table[0]
		for i, s := range row.cols {
			switch i {
			case 0:
				fmt.Fprintf(w, "%-*s", max[i], s)
			default:
				fmt.Fprintf(w, "  %-*s", max[i], s)
			case len(row.cols) - 1:
				fmt.Fprintf(w, "  %s\n", s)
			}
		}

		// data
		for _, row := range table[1:] {
			for i, s := range row.cols {
				switch {
				case len(row.cols) == 1:
					// Header row
					fmt.Fprint(w, s)
				case i == 0:
					fmt.Fprintf(w, "%-*s", max[i], s)
				default:
					if i == len(row.cols)-1 && len(s) > 0 && s[0] == '(' {
						// Left-align p value.
						fmt.Fprintf(w, "  %s", s)
						break
					}
					fmt.Fprintf(w, "  %*s", max[i], s)
				}
			}
			fmt.Fprintf(w, "\n")
		}
	}
}

func must(e error) {
	if e != nil {
		panic(e)
	}
}

// FormatCSV appends a CSV formatting of the tables to w.
// norange suppresses the range columns.
func FormatCSV(w io.Writer, tables []*Table, norange bool) {
	var textTables [][]*textRow
	for _, t := range tables {
		textTables = append(textTables, toCSV(t, norange))
	}

	for i, table := range textTables {
		if i > 0 {
			fmt.Fprintf(w, "\n")
		}
		csvw := csv.NewWriter(w)

		// headings
		row := table[0]
		must(csvw.Write(row.cols))

		// data
		for _, row := range table[1:] {
			must(csvw.Write(row.cols))
		}
		csvw.Flush()
	}
}

// trimCommonPathPrefix returns a string slice with common
// path-separator-terminated prefixes removed.  Empty strings
// are ignored and a singleton non-empty string is left unchanged
func trimCommonPathPrefix(ss []string) []string {
	commonPrefixLen := -1
	commonPrefix := ""
	trimCommonPrefix := false // true if more than one data row w/ non-empty title

	// begin finding common byte prefix (could end on partial rune)
	for _, s := range ss {
		if s == "" {
			continue
		}
		if commonPrefixLen == -1 {
			commonPrefix = s
			commonPrefixLen = len(s)
			continue
		}
		trimCommonPrefix = true // More than one not-empty
		if commonPrefixLen > len(s) {
			commonPrefixLen = len(s)
			commonPrefix = commonPrefix[0:commonPrefixLen]
		}
		for j := 0; j < commonPrefixLen; j++ {
			if commonPrefix[j] != s[j] {
				commonPrefixLen = j
				commonPrefix = commonPrefix[0:commonPrefixLen]
				break
			}
		}
	}
	if !trimCommonPrefix || commonPrefixLen == 0 {
		return ss
	}
	// trim to include last path separator.
	commonPrefixLen = 1 + strings.LastIndex(commonPrefix, string(filepath.Separator))
	if commonPrefixLen == 0 {
		// No separator found means commonPrefixLen = 1 + (-1) == 0
		return ss
	}

	rs := []string{}
	for _, s := range ss {
		if s == "" {
			rs = append(rs, "")
		} else {
			rs = append(rs, s[commonPrefixLen:])
		}
	}
	return rs
}

// A textRow is a row of printed text columns.
type textRow struct {
	cols []string
}

func newTextRow(cols ...string) *textRow {
	return &textRow{cols: cols}
}

// newTextRowDelta returns a labeled row of text, with "±" inserted after
// each member of "cols" unless norange is true.
func newTextRowDelta(norange bool, label string, cols ...string) *textRow {
	newcols := []string{}
	newcols = append(newcols, label)
	for _, s := range cols {
		newcols = append(newcols, s)
		if !norange {
			newcols = append(newcols, "±")
		}
	}
	return &textRow{cols: newcols}
}

func (r *textRow) add(col string) {
	r.cols = append(r.cols, col)
}

func (r *textRow) trim() {
	for len(r.cols) > 0 && r.cols[len(r.cols)-1] == "" {
		r.cols = r.cols[:len(r.cols)-1]
	}
}

// toText converts the Table to a textual grid of cells,
// which can then be printed in fixed-width output.
func toText(t *Table) []*textRow {
	var textRows []*textRow
	switch len(t.Configs) {
	case 1:
		textRows = append(textRows, newTextRow("name", t.Metric))
	case 2:
		textRows = append(textRows, newTextRow("name", "old "+t.Metric, "new "+t.Metric, "delta"))
	default:
		row := newTextRow("name \\ " + t.Metric)
		row.cols = append(row.cols, t.Configs...) // TODO Should this also trim common path prefix? (see toCSV)
		textRows = append(textRows, row)
	}

	var group string

	for _, row := range t.Rows {
		if row.Group != group {
			group = row.Group
			textRows = append(textRows, newTextRow(group))
		}
		text := newTextRow(row.Benchmark)
		for _, m := range row.Metrics {
			text.cols = append(text.cols, m.Format(row.Scaler))
		}
		if len(t.Configs) == 2 {
			delta := row.Delta
			if delta == "~" {
				delta = "~   "
			}
			text.cols = append(text.cols, delta)
			text.cols = append(text.cols, row.Note)
		}
		textRows = append(textRows, text)
	}
	for _, r := range textRows {
		r.trim()
	}
	return textRows
