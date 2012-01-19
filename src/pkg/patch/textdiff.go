// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package patch

import (
	"bytes"
	"errors"
)

type TextDiff []TextChunk

// A TextChunk specifies an edit to a section of a file:
// the text beginning at Line, which should be exactly Old,
// is to be replaced with New.
type TextChunk struct {
	Line int
	Old  []byte
	New  []byte
}

func ParseTextDiff(raw []byte) (TextDiff, error) {
	var chunkHeader []byte

	// Copy raw so it is safe to keep references to slices.
	_, chunks := sections(raw, "@@ -")
	delta := 0
	diff := make(TextDiff, len(chunks))
	for i, raw := range chunks {
		c := &diff[i]

		// Parse start line: @@ -oldLine,oldCount +newLine,newCount @@ junk
		chunk := splitLines(raw)
		chunkHeader = chunk[0]
		var ok bool
		var oldLine, oldCount, newLine, newCount int
		s := chunkHeader
		if oldLine, s, ok = atoi(s, "@@ -", 10); !ok {
			goto ErrChunkHdr
		}
		if len(s) == 0 || s[0] != ',' {
			oldCount = 1
		} else if oldCount, s, ok = atoi(s, ",", 10); !ok {
			goto ErrChunkHdr
		}
		if newLine, s, ok = atoi(s, " +", 10); !ok {
			goto ErrChunkHdr
		}
		if len(s) == 0 || s[0] != ',' {
			newCount = 1
		} else if newCount, s, ok = atoi(s, ",", 10); !ok {
			goto ErrChunkHdr
		}
		if !hasPrefix(s, " @@") {
			goto ErrChunkHdr
		}

		// Special case: for created or deleted files, the empty half
		// is given as starting at line 0.  Translate to line 1.
		if oldCount == 0 && oldLine == 0 {
			oldLine = 1
		}
		if newCount == 0 && newLine == 0 {
			newLine = 1
		}

		// Count lines in text
		var dropOldNL, dropNewNL bool
		var nold, nnew int
		var lastch byte
		chunk = chunk[1:]
		for _, l := range chunk {
			if nold == oldCount && nnew == newCount && (len(l) == 0 || l[0] != '\\') {
				if len(bytes.TrimSpace(l)) != 0 {
					return nil, SyntaxError("too many chunk lines")
				}
				continue
			}
			if len(l) == 0 {
				return nil, SyntaxError("empty chunk line")
			}
			switch l[0] {
			case '+':
				nnew++
			case '-':
				nold++
			case ' ':
				nnew++
				nold++
			case '\\':
				if _, ok := skip(l, "\\ No newline at end of file"); ok {
					switch lastch {
					case '-':
						dropOldNL = true
					case '+':
						dropNewNL = true
					case ' ':
						dropOldNL = true
						dropNewNL = true
					default:
						return nil, SyntaxError("message `\\ No newline at end of file' out of context")
					}
					break
				}
				fallthrough
			default:
				return nil, SyntaxError("unexpected chunk line: " + string(l))
			}
			lastch = l[0]
		}

		// Does it match the header?
		if nold != oldCount || nnew != newCount {
			return nil, SyntaxError("chunk header does not match line count: " + string(chunkHeader))
		}
		if oldLine+delta != newLine {
			return nil, SyntaxError("chunk delta is out of sync with previous chunks")
		}
		delta += nnew - nold
		c.Line = oldLine

		var old, new bytes.Buffer
		nold = 0
		nnew = 0
		for _, l := range chunk {
			if nold == oldCount && nnew == newCount {
				break
			}
			ch, l := l[0], l[1:]
			if ch == '\\' {
				continue
			}
			if ch != '+' {
				old.Write(l)
				nold++
			}
			if ch != '-' {
				new.Write(l)
				nnew++
			}
		}
		c.Old = old.Bytes()
		c.New = new.Bytes()
		if dropOldNL {
			c.Old = c.Old[0 : len(c.Old)-1]
		}
		if dropNewNL {
			c.New = c.New[0 : len(c.New)-1]
		}
	}
	return diff, nil

ErrChunkHdr:
	return nil, SyntaxError("unexpected chunk header line: " + string(chunkHeader))
}

var ErrPatchFailure = errors.New("patch did not apply cleanly")

// Apply applies the changes listed in the diff
// to the data, returning the new version.
func (d TextDiff) Apply(data []byte) ([]byte, error) {
	var buf bytes.Buffer
	line := 1
	for _, c := range d {
		var ok bool
		var prefix []byte
		prefix, data, ok = getLine(data, c.Line-line)
		if !ok || !bytes.HasPrefix(data, c.Old) {
			return nil, ErrPatchFailure
		}
		buf.Write(prefix)
		data = data[len(c.Old):]
		buf.Write(c.New)
		line = c.Line + bytes.Count(c.Old, newline)
	}
	buf.Write(data)
	return buf.Bytes(), nil
}
