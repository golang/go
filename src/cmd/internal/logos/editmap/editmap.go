// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package editmap

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

const Suffix = ".edits"

const Version = 2

type Segment struct {
	Out  int `json:"out"`
	Len  int `json:"len"`
	Orig int `json:"orig"`
}

type EditSpan struct {
	Start int `json:"start"`
	End   int `json:"end"`
	Len   int `json:"len"`
}

type EditMap struct {
	Version  int        `json:"version"`
	Source   string     `json:"source"`
	Size     int        `json:"size"`
	Out      int        `json:"out"`
	Edits    []EditSpan `json:"edits"`
	Segments []Segment  `json:"segments"`
}

func Write(path, source string, size, out int, spans []EditSpan, segments []Segment) error {
	data, err := json.Marshal(EditMap{
		Version:  Version,
		Source:   filepath.Base(source),
		Size:     size,
		Out:      out,
		Edits:    spans,
		Segments: segments,
	})
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0o666)
}

type Edge int

const (
	Leading Edge = iota
	Trailing
)

type Map struct {
	SrcLen   int
	OutLen   int
	Segments []Segment
}

func New(src, out []byte, m EditMap) (*Map, error) {
	if err := Check(src, out, m); err != nil {
		return nil, err
	}
	return &Map{SrcLen: len(src), OutLen: len(out), Segments: m.Segments}, nil
}

func Check(src, out []byte, m EditMap) error {
	if m.Version != Version {
		return fmt.Errorf("edit map for %s has unsupported version %d", m.Source, m.Version)
	}
	if m.Size != len(src) || m.Out != len(out) {
		return fmt.Errorf("edit map for %s does not match its texts", m.Source)
	}
	prev, delta := 0, 0
	for _, s := range m.Edits {
		if !s.fits(prev, len(src)) {
			return fmt.Errorf("edit map for %s holds the invalid span %d:%d", m.Source, s.Start, s.End)
		}
		prev = s.End
		delta += s.Len - (s.End - s.Start)
	}
	if len(src)+delta != len(out) {
		return fmt.Errorf("edit map for %s does not add up to the lowered size", m.Source)
	}
	next := 0
	for _, s := range m.Segments {
		if s.Len <= 0 || s.Out != next {
			return fmt.Errorf("edit map for %s has segments that do not partition the lowered text", m.Source)
		}
		if s.Len > len(out)-s.Out {
			return fmt.Errorf("edit map for %s has segments that do not cover the lowered text", m.Source)
		}
		end := s.Out + s.Len
		if s.Orig < -1 {
			return fmt.Errorf("edit map for %s holds the invalid segment %d:%d", m.Source, s.Out, end)
		}
		if s.Orig >= 0 {
			if s.Len > len(src)-s.Orig {
				return fmt.Errorf("edit map for %s holds the copy segment %d:%d outside its source", m.Source, s.Out, end)
			}
			if !bytes.Equal(src[s.Orig:s.Orig+s.Len], out[s.Out:end]) {
				return fmt.Errorf("edit map for %s holds the copy segment %d:%d that differs from its source", m.Source, s.Out, end)
			}
		}
		next = end
	}
	if next != len(out) {
		return fmt.Errorf("edit map for %s has segments that do not cover the lowered text", m.Source)
	}
	return nil
}

func (s EditSpan) fits(prev, size int) bool {
	return s.Start >= prev && s.End >= s.Start && s.End <= size && s.Len >= 0
}

func (m *Map) LoweredOffset(o int, e Edge) (int, bool) {
	if m.SrcLen == 0 && o == 0 {
		if e == Trailing {
			return 0, true
		}
		return m.OutLen, true
	}
	if o == 0 && e == Trailing {
		for _, s := range m.Segments {
			if s.Orig >= 0 {
				if s.Orig == 0 && s.Out > 0 {
					return 0, true
				}
				break
			}
		}
	}
	for i, s := range m.Segments {
		if s.Orig < 0 {
			continue
		}
		if o >= s.Orig && o < s.Orig+s.Len {
			return s.Out + o - s.Orig, true
		}
		if o != s.Orig+s.Len {
			continue
		}
		if e == Trailing {
			return s.Out + s.Len, true
		}
		if i+1 < len(m.Segments) && m.Segments[i+1].Orig < 0 {
			g := m.Segments[i+1]
			pure := (i+2 >= len(m.Segments) && o == m.SrcLen) ||
				(i+2 < len(m.Segments) && m.Segments[i+2].Orig == o)
			if pure {
				return g.Out + g.Len, true
			}
		}
		return s.Out + s.Len, true
	}
	return 0, false
}

func (m *Map) LoweredOffsetNear(o int, e Edge) int {
	if v, ok := m.LoweredOffset(o, e); ok {
		return v
	}
	if e == Trailing {
		bestOrig := -1
		bestOut := m.OutLen
		for _, s := range m.Segments {
			if s.Orig < 0 || s.Orig <= o {
				continue
			}
			if bestOrig == -1 || s.Orig < bestOrig || (s.Orig == bestOrig && s.Out < bestOut) {
				bestOrig = s.Orig
				bestOut = s.Out
			}
		}
		return bestOut
	}
	bestEnd := -1
	bestOut := 0
	for _, s := range m.Segments {
		if s.Orig < 0 {
			continue
		}
		end := s.Orig + s.Len
		if end > o {
			continue
		}
		outEnd := s.Out + s.Len
		if bestEnd == -1 || end > bestEnd || (end == bestEnd && outEnd < bestOut) {
			bestEnd = end
			bestOut = outEnd
		}
	}
	return bestOut
}

func (m *Map) genAnchors(i, prevEnd int) (start, end int) {
	start, end = prevEnd, m.SrcLen
	for _, s := range m.Segments[i+1:] {
		if s.Orig >= 0 {
			end = s.Orig
			break
		}
	}
	return start, end
}

func (m *Map) OriginalOffset(o int, e Edge) (int, bool) {
	if o == m.OutLen && len(m.Segments) > 0 {
		last := m.Segments[len(m.Segments)-1]
		if last.Orig < 0 {
			return m.SrcLen, false
		}
		if end := last.Orig + last.Len; end == m.SrcLen {
			return end, false
		} else if e == Trailing {
			return end, false
		} else {
			return m.SrcLen, false
		}
	}
	prevEnd := 0
	for i, s := range m.Segments {
		end := s.Out + s.Len
		if o < s.Out || o >= end {
			if s.Orig >= 0 {
				prevEnd = s.Orig + s.Len
			}
			continue
		}
		if s.Orig >= 0 {
			if o == s.Out && s.Orig > prevEnd && e == Trailing {
				return prevEnd, false
			}
			return s.Orig + o - s.Out, false
		}
		if o == s.Out {
			return prevEnd, false
		}
		start, finish := m.genAnchors(i, prevEnd)
		if e == Trailing {
			return finish, true
		}
		return start, true
	}
	return m.SrcLen, false
}

func (m *Map) WidenOverGenerated(loStart, loEnd, start, end int) (int, int) {
	prevEnd := 0
	for i, s := range m.Segments {
		segEnd := s.Out + s.Len
		if s.Orig >= 0 {
			prevEnd = s.Orig + s.Len
			continue
		}
		if s.Out >= loEnd || segEnd <= loStart {
			continue
		}
		aStart, aFinish := m.genAnchors(i, prevEnd)
		if aStart > aFinish {
			continue
		}
		start = min(start, aStart)
		end = max(end, aFinish)
	}
	return start, end
}
