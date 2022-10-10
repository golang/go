// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package lsppos provides utilities for working with LSP positions. Much of
// this functionality is duplicated from the internal/span package, but this
// package is simpler and more accurate with respect to newline terminated
// content.
//
// See https://microsoft.github.io/language-server-protocol/specification#textDocuments
// for a description of LSP positions. Notably:
//   - Positions are specified by a 0-based line count and 0-based utf-16
//     character offset.
//   - Positions are line-ending agnostic: there is no way to specify \r|\n or
//     \n|. Instead the former maps to the end of the current line, and the
//     latter to the start of the next line.
package lsppos

import (
	"bytes"
	"errors"
	"sort"
	"unicode/utf8"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
)

// Mapper maps utf-8 byte offsets to LSP positions for a single file.
type Mapper struct {
	nonASCII bool
	content  []byte

	// Start-of-line positions. If src is newline-terminated, the final entry
	// will be len(content).
	lines []int
}

// NewMapper creates a new Mapper for the given content.
func NewMapper(content []byte) *Mapper {
	nlines := bytes.Count(content, []byte("\n"))
	m := &Mapper{
		content: content,
		lines:   make([]int, 1, nlines+1), // initially []int{0}
	}
	for offset, b := range content {
		if b == '\n' {
			m.lines = append(m.lines, offset+1)
		}
		if b >= utf8.RuneSelf {
			m.nonASCII = true
		}
	}
	return m
}

// LineColUTF16 returns the 0-based UTF-16 line and character index for the
// given offset. It returns -1, -1 if offset is out of bounds for the file
// being mapped.
func (m *Mapper) LineColUTF16(offset int) (line, char int) {
	if offset < 0 || offset > len(m.content) {
		return -1, -1
	}
	nextLine := sort.Search(len(m.lines), func(i int) bool {
		return offset < m.lines[i]
	})
	if nextLine == 0 {
		return -1, -1
	}
	line = nextLine - 1
	start := m.lines[line]
	var charOffset int
	if m.nonASCII {
		charOffset = UTF16len(m.content[start:offset])
	} else {
		charOffset = offset - start
	}

	var eol int
	if line == len(m.lines)-1 {
		eol = len(m.content)
	} else {
		eol = m.lines[line+1] - 1
	}

	// Adjustment for line-endings: \r|\n is the same as |\r\n.
	if offset == eol && offset > 0 && m.content[offset-1] == '\r' {
		charOffset--
	}

	return line, charOffset
}

// Position returns the protocol position corresponding to the given offset. It
// returns false if offset is out of bounds for the file being mapped.
func (m *Mapper) Position(offset int) (protocol.Position, bool) {
	l, c := m.LineColUTF16(offset)
	if l < 0 {
		return protocol.Position{}, false
	}
	return protocol.Position{
		Line:      uint32(l),
		Character: uint32(c),
	}, true
}

// Range returns the protocol range corresponding to the given start and end
// offsets.
func (m *Mapper) Range(start, end int) (protocol.Range, error) {
	startPos, ok := m.Position(start)
	if !ok {
		return protocol.Range{}, errors.New("invalid start position")
	}
	endPos, ok := m.Position(end)
	if !ok {
		return protocol.Range{}, errors.New("invalid end position")
	}

	return protocol.Range{Start: startPos, End: endPos}, nil
}

// UTF16len returns the UTF-16 length of the UTF-8 encoded content, were it to
// be re-encoded as UTF-16.
func UTF16len(buf []byte) int {
	// This function copies buf, but microbenchmarks showed it to be faster than
	// using utf8.DecodeRune due to inlining and avoiding bounds checks.
	cnt := 0
	for _, r := range string(buf) {
		cnt++
		if r >= 1<<16 {
			cnt++
		}
	}
	return cnt
}
