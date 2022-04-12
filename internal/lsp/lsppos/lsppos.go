// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package lsppos provides utilities for working with LSP positions.
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
	"sort"
	"unicode/utf8"
)

type Mapper struct {
	nonASCII bool
	src      []byte

	// Start-of-line positions. If src is newline-terminated, the final entry will be empty.
	lines []int
}

func NewMapper(src []byte) *Mapper {
	m := &Mapper{src: src}
	if len(src) == 0 {
		return m
	}
	m.lines = []int{0}
	for offset, b := range src {
		if b == '\n' {
			m.lines = append(m.lines, offset+1)
		}
		if b >= utf8.RuneSelf {
			m.nonASCII = true
		}
	}
	return m
}

func (m *Mapper) Position(offset int) (line, char int) {
	if offset < 0 || offset > len(m.src) {
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
		charOffset = UTF16len(m.src[start:offset])
	} else {
		charOffset = offset - start
	}

	var eol int
	if line == len(m.lines)-1 {
		eol = len(m.src)
	} else {
		eol = m.lines[line+1] - 1
	}

	// Adjustment for line-endings: \r|\n is the same as |\r\n.
	if offset == eol && offset > 0 && m.src[offset-1] == '\r' {
		charOffset--
	}

	return line, charOffset
}

func UTF16len(buf []byte) int {
	cnt := 0
	for _, r := range string(buf) {
		cnt++
		if r >= 1<<16 {
			cnt++
		}
	}
	return cnt
}
