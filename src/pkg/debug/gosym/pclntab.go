// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * Line tables
 */

package gosym

import "encoding/binary"

type LineTable struct {
	Data []byte
	PC   uint64
	Line int
}

// TODO(rsc): Need to pull in quantum from architecture definition.
const quantum = 1

func (t *LineTable) parse(targetPC uint64, targetLine int) (b []byte, pc uint64, line int) {
	// The PC/line table can be thought of as a sequence of
	//  <pc update>* <line update>
	// batches.  Each update batch results in a (pc, line) pair,
	// where line applies to every PC from pc up to but not
	// including the pc of the next pair.
	//
	// Here we process each update individually, which simplifies
	// the code, but makes the corner cases more confusing.
	b, pc, line = t.Data, t.PC, t.Line
	for pc <= targetPC && line != targetLine && len(b) > 0 {
		code := b[0]
		b = b[1:]
		switch {
		case code == 0:
			if len(b) < 4 {
				b = b[0:0]
				break
			}
			val := binary.BigEndian.Uint32(b)
			b = b[4:]
			line += int(val)
		case code <= 64:
			line += int(code)
		case code <= 128:
			line -= int(code - 64)
		default:
			pc += quantum * uint64(code-128)
			continue
		}
		pc += quantum
	}
	return b, pc, line
}

func (t *LineTable) slice(pc uint64) *LineTable {
	data, pc, line := t.parse(pc, -1)
	return &LineTable{data, pc, line}
}

func (t *LineTable) PCToLine(pc uint64) int {
	_, _, line := t.parse(pc, -1)
	return line
}

func (t *LineTable) LineToPC(line int, maxpc uint64) uint64 {
	_, pc, line1 := t.parse(maxpc, line)
	if line1 != line {
		return 0
	}
	// Subtract quantum from PC to account for post-line increment
	return pc - quantum
}

// NewLineTable returns a new PC/line table
// corresponding to the encoded data.
// Text must be the start address of the
// corresponding text segment.
func NewLineTable(data []byte, text uint64) *LineTable {
	return &LineTable{data, text, 0}
}
