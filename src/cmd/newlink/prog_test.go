// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"testing"
)

// shiftProg adjusts the addresses in p.
// It adds vdelta to all virtual addresses and fdelta to all file offsets.
func shiftProg(p *Prog, vdelta Addr, fdelta Addr) {
	p.Entry += vdelta
	for _, seg := range p.Segments {
		seg.FileOffset += fdelta
		seg.VirtAddr += vdelta
		for _, sect := range seg.Sections {
			sect.VirtAddr += vdelta
			for _, sym := range sect.Syms {
				sym.Addr += vdelta
			}
		}
	}
}

// diffProg returns a list of differences between p and q,
// assuming p is being checked and q is the correct answer.
func diffProg(p, q *Prog) []string {
	var errors []string
	if p.UnmappedSize != q.UnmappedSize {
		errors = append(errors, fmt.Sprintf("p.UnmappedSize = %#x, want %#x", p.UnmappedSize, q.UnmappedSize))
	}
	if p.HeaderSize != q.HeaderSize {
		errors = append(errors, fmt.Sprintf("p.HeaderSize = %#x, want %#x", p.HeaderSize, q.HeaderSize))
	}
	if p.Entry != q.Entry {
		errors = append(errors, fmt.Sprintf("p.Entry = %#x, want %#x", p.Entry, q.Entry))
	}
	for i := 0; i < len(p.Segments) || i < len(q.Segments); i++ {
		if i >= len(p.Segments) {
			errors = append(errors, fmt.Sprintf("p missing segment %q", q.Segments[i].Name))
			continue
		}
		if i >= len(q.Segments) {
			errors = append(errors, fmt.Sprintf("p has extra segment %q", p.Segments[i].Name))
			continue
		}
		pseg := p.Segments[i]
		qseg := q.Segments[i]
		if pseg.Name != qseg.Name {
			errors = append(errors, fmt.Sprintf("segment %d Name = %q, want %q", i, pseg.Name, qseg.Name))
			continue // probably out of sync
		}
		if pseg.VirtAddr != qseg.VirtAddr {
			errors = append(errors, fmt.Sprintf("segment %q VirtAddr = %#x, want %#x", pseg.Name, pseg.VirtAddr, qseg.VirtAddr))
		}
		if pseg.VirtSize != qseg.VirtSize {
			errors = append(errors, fmt.Sprintf("segment %q VirtSize = %#x, want %#x", pseg.Name, pseg.VirtSize, qseg.VirtSize))
		}
		if pseg.FileOffset != qseg.FileOffset {
			errors = append(errors, fmt.Sprintf("segment %q FileOffset = %#x, want %#x", pseg.Name, pseg.FileOffset, qseg.FileOffset))
		}
		if pseg.FileSize != qseg.FileSize {
			errors = append(errors, fmt.Sprintf("segment %q FileSize = %#x, want %#x", pseg.Name, pseg.FileSize, qseg.FileSize))
		}
		if len(pseg.Data) != len(qseg.Data) {
			errors = append(errors, fmt.Sprintf("segment %q len(Data) = %d, want %d", pseg.Name, len(pseg.Data), len(qseg.Data)))
		} else if !bytes.Equal(pseg.Data, qseg.Data) {
			errors = append(errors, fmt.Sprintf("segment %q Data mismatch:\n\thave %x\n\twant %x", pseg.Name, pseg.Data, qseg.Data))
		}

		for j := 0; j < len(pseg.Sections) || j < len(qseg.Sections); j++ {
			if j >= len(pseg.Sections) {
				errors = append(errors, fmt.Sprintf("segment %q missing section %q", pseg.Name, qseg.Sections[i].Name))
				continue
			}
			if j >= len(qseg.Sections) {
				errors = append(errors, fmt.Sprintf("segment %q has extra section %q", pseg.Name, pseg.Sections[i].Name))
				continue
			}
			psect := pseg.Sections[j]
			qsect := qseg.Sections[j]
			if psect.Name != qsect.Name {
				errors = append(errors, fmt.Sprintf("segment %q, section %d Name = %q, want %q", pseg.Name, j, psect.Name, qsect.Name))
				continue // probably out of sync
			}

			if psect.VirtAddr != qsect.VirtAddr {
				errors = append(errors, fmt.Sprintf("segment %q section %q VirtAddr = %#x, want %#x", pseg.Name, psect.Name, psect.VirtAddr, qsect.VirtAddr))
			}
			if psect.Size != qsect.Size {
				errors = append(errors, fmt.Sprintf("segment %q section %q Size = %#x, want %#x", pseg.Name, psect.Name, psect.Size, qsect.Size))
			}
			if psect.Align != qsect.Align {
				errors = append(errors, fmt.Sprintf("segment %q section %q Align = %#x, want %#x", pseg.Name, psect.Name, psect.Align, qsect.Align))
			}
		}
	}

	return errors
}

// cloneProg returns a deep copy of p.
func cloneProg(p *Prog) *Prog {
	q := new(Prog)
	*q = *p
	q.Segments = make([]*Segment, len(p.Segments))
	for i, seg := range p.Segments {
		q.Segments[i] = cloneSegment(seg)
	}
	return q
}

// cloneSegment returns a deep copy of seg.
func cloneSegment(seg *Segment) *Segment {
	t := new(Segment)
	*t = *seg
	t.Sections = make([]*Section, len(seg.Sections))
	for i, sect := range seg.Sections {
		t.Sections[i] = cloneSection(sect)
	}
	t.Data = make([]byte, len(seg.Data))
	copy(t.Data, seg.Data)
	return t
}

// cloneSection returns a deep copy of section.
func cloneSection(sect *Section) *Section {
	// At the moment, there's nothing we need to make a deep copy of.
	t := new(Section)
	*t = *sect
	return t
}

const saveMismatch = true

// checkGolden checks that data matches the named file.
// If not, it reports the error to the test.
func checkGolden(t *testing.T, data []byte, name string) {
	golden := mustParseHexdumpFile(t, name)
	if !bytes.Equal(data, golden) {
		if saveMismatch {
			ioutil.WriteFile(name+".raw", data, 0666)
			ioutil.WriteFile(name+".hex", []byte(hexdump(data)), 0666)
		}
		// TODO(rsc): A better diff would be nice, as needed.
		i := 0
		for i < len(data) && i < len(golden) && data[i] == golden[i] {
			i++
		}
		if i >= len(data) {
			t.Errorf("%s: output file shorter than expected: have %d bytes, want %d", name, len(data), len(golden))
		} else if i >= len(golden) {
			t.Errorf("%s: output file larger than expected: have %d bytes, want %d", name, len(data), len(golden))
		} else {
			t.Errorf("%s: output file differs at byte %d: have %#02x, want %#02x", name, i, data[i], golden[i])
		}
	}
}
