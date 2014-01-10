// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Mach-O (Darwin) object file writing.

package main

import (
	"debug/macho"
	"encoding/binary"
	"io"
	"strings"
)

// machoFormat is the implementation of formatter.
type machoFormat struct{}

// machoHeader and friends are data structures
// corresponding to the Mach-O file header
// to be written to disk.

const (
	macho64Bit     = 1 << 24
	machoSubCPU386 = 3
)

// machoArch describes a Mach-O target architecture.
type machoArch struct {
	CPU    uint32
	SubCPU uint32
}

// machoHeader is the Mach-O file header.
type machoHeader struct {
	machoArch
	FileType uint32
	Loads    []*machoLoad
	Segments []*machoSegment
	p        *Prog // for reporting errors
}

// machoLoad is a Mach-O load command.
type machoLoad struct {
	Type uint32
	Data []uint32
}

// machoSegment is a Mach-O segment.
type machoSegment struct {
	Name       string
	VirtAddr   Addr
	VirtSize   Addr
	FileOffset Addr
	FileSize   Addr
	Prot1      uint32
	Prot2      uint32
	Flags      uint32
	Sections   []*machoSection
}

// machoSection is a Mach-O section, inside a segment.
type machoSection struct {
	Name    string
	Segment string
	Addr    Addr
	Size    Addr
	Offset  uint32
	Align   uint32
	Reloc   uint32
	Nreloc  uint32
	Flags   uint32
	Res1    uint32
	Res2    uint32
}

// layout positions the segments and sections in p
// to make room for the Mach-O file header.
// That is, it edits their VirtAddr fields to adjust for the presence
// of the Mach-O header at the beginning of the address space.
func (machoFormat) headerSize(p *Prog) (virt, file Addr) {
	var h machoHeader
	h.init(p)
	size := Addr(h.size())
	size = round(size, 4096)
	p.HeaderSize = size
	return size, size
}

// write writes p to w as a Mach-O executable.
// layout(p) must have already been called,
// and the number, sizes, and addresses of the segments
// and sections must not have been modified since the call.
func (machoFormat) write(w io.Writer, p *Prog) {
	var h machoHeader
	h.init(p)
	off := Addr(0)
	enc := h.encode()
	w.Write(enc)
	off += Addr(len(enc))
	for _, seg := range p.Segments {
		if seg.FileOffset < off {
			h.p.errorf("mach-o error: invalid file offset")
		}
		w.Write(make([]byte, int(seg.FileOffset-off)))
		if seg.FileSize != Addr(len(seg.Data)) {
			h.p.errorf("mach-o error: invalid file size")
		}
		w.Write(seg.Data)
		off = seg.FileOffset + Addr(len(seg.Data))
	}
}

// Conversion of Prog to macho data structures.

// machoArches maps from GOARCH to machoArch.
var machoArches = map[string]machoArch{
	"amd64": {
		CPU:    uint32(macho.CpuAmd64),
		SubCPU: uint32(machoSubCPU386),
	},
}

// init initializes the header h to describe p.
func (h *machoHeader) init(p *Prog) {
	h.p = p
	h.Segments = nil
	h.Loads = nil
	var ok bool
	h.machoArch, ok = machoArches[p.GOARCH]
	if !ok {
		p.errorf("mach-o: unknown target GOARCH %q", p.GOARCH)
		return
	}
	h.FileType = uint32(macho.TypeExec)

	mseg := h.addSegment(p, "__PAGEZERO", nil)
	mseg.VirtSize = p.UnmappedSize

	for _, seg := range p.Segments {
		h.addSegment(p, "__"+strings.ToUpper(seg.Name), seg)
	}

	var data []uint32
	switch h.CPU {
	default:
		p.errorf("mach-o: unknown cpu %#x for GOARCH %q", h.CPU, p.GOARCH)
	case uint32(macho.CpuAmd64):
		data = make([]uint32, 2+42)
		data[0] = 4                  // thread type
		data[1] = 42                 // word count
		data[2+32] = uint32(p.Entry) // RIP register, in two parts
		data[2+32+1] = uint32(p.Entry >> 32)
	}

	h.Loads = append(h.Loads, &machoLoad{
		Type: uint32(macho.LoadCmdUnixThread),
		Data: data,
	})
}

// addSegment adds to h a Mach-O segment like seg with the given name.
func (h *machoHeader) addSegment(p *Prog, name string, seg *Segment) *machoSegment {
	mseg := &machoSegment{
		Name: name,
	}
	h.Segments = append(h.Segments, mseg)
	if seg == nil {
		return mseg
	}

	mseg.VirtAddr = seg.VirtAddr
	mseg.VirtSize = seg.VirtSize
	mseg.FileOffset = round(seg.FileOffset, 4096)
	mseg.FileSize = seg.FileSize

	if name == "__TEXT" {
		// Initially RWX, then just RX
		mseg.Prot1 = 7
		mseg.Prot2 = 5

		// Text segment maps Mach-O header, needed by dynamic linker.
		mseg.VirtAddr -= p.HeaderSize
		mseg.VirtSize += p.HeaderSize
		mseg.FileOffset -= p.HeaderSize
		mseg.FileSize += p.HeaderSize
	} else {
		// RW
		mseg.Prot1 = 3
		mseg.Prot2 = 3
	}

	for _, sect := range seg.Sections {
		h.addSection(mseg, seg, sect)
	}
	return mseg
}

// addSection adds to mseg a Mach-O section like sect, inside seg, with the given name.
func (h *machoHeader) addSection(mseg *machoSegment, seg *Segment, sect *Section) {
	msect := &machoSection{
		Name:    "__" + sect.Name,
		Segment: mseg.Name,
		// Reloc: sect.RelocOffset,
		// NumReloc: sect.RelocLen / 8,
		Addr: sect.VirtAddr,
		Size: sect.Size,
	}
	mseg.Sections = append(mseg.Sections, msect)

	for 1<<msect.Align < sect.Align {
		msect.Align++
	}

	if off := sect.VirtAddr - seg.VirtAddr; off < seg.FileSize {
		// Data in file.
		if sect.Size > seg.FileSize-off {
			h.p.errorf("mach-o error: section crosses file boundary")
		}
		msect.Offset = uint32(seg.FileOffset + off)
	} else {
		// Zero filled.
		msect.Flags |= 1
	}

	if sect.Name == "text" {
		msect.Flags |= 0x400 // contains executable instructions
	}
}

// A machoWriter helps write Mach-O headers.
// It is basically a buffer with some helper routines for writing integers.
type machoWriter struct {
	dst   []byte
	tmp   [8]byte
	order binary.ByteOrder
	is64  bool
	p     *Prog
}

// if64 returns x if w is writing a 64-bit object file; otherwise it returns y.
func (w *machoWriter) if64(x, y interface{}) interface{} {
	if w.is64 {
		return x
	}
	return y
}

// encode encodes each of the given arguments into the writer.
// It encodes uint32, []uint32, uint64, and []uint64 by writing each value
// in turn in the correct byte order for the output file.
// It encodes an Addr as a uint64 if writing a 64-bit output file, or else as a uint32.
// It encodes []byte and string by writing the raw bytes (no length prefix).
// It skips nil values in the args list.
func (w *machoWriter) encode(args ...interface{}) {
	for _, arg := range args {
		switch arg := arg.(type) {
		default:
			w.p.errorf("mach-o error: cannot encode %T", arg)
		case nil:
			// skip
		case []byte:
			w.dst = append(w.dst, arg...)
		case string:
			w.dst = append(w.dst, arg...)
		case uint32:
			w.order.PutUint32(w.tmp[:], arg)
			w.dst = append(w.dst, w.tmp[:4]...)
		case []uint32:
			for _, x := range arg {
				w.order.PutUint32(w.tmp[:], x)
				w.dst = append(w.dst, w.tmp[:4]...)
			}
		case uint64:
			w.order.PutUint64(w.tmp[:], arg)
			w.dst = append(w.dst, w.tmp[:8]...)
		case Addr:
			if w.is64 {
				w.order.PutUint64(w.tmp[:], uint64(arg))
				w.dst = append(w.dst, w.tmp[:8]...)
			} else {
				if Addr(uint32(arg)) != arg {
					w.p.errorf("mach-o error: truncating address %#x to uint32", arg)
				}
				w.order.PutUint32(w.tmp[:], uint32(arg))
				w.dst = append(w.dst, w.tmp[:4]...)
			}
		}
	}
}

// segmentSize returns the size of the encoding of seg in bytes.
func (w *machoWriter) segmentSize(seg *machoSegment) int {
	if w.is64 {
		return 18*4 + 20*4*len(seg.Sections)
	}
	return 14*4 + 22*4*len(seg.Sections)
}

// zeroPad returns the string s truncated or padded with NULs to n bytes.
func zeroPad(s string, n int) string {
	if len(s) >= n {
		return s[:n]
	}
	return s + strings.Repeat("\x00", n-len(s))
}

// size returns the encoded size of the header.
func (h *machoHeader) size() int {
	// Could write separate code, but encoding is cheap; encode and throw it away.
	return len(h.encode())
}

// encode returns the Mach-O encoding of the header.
func (h *machoHeader) encode() []byte {
	w := &machoWriter{p: h.p}
	w.is64 = h.CPU&macho64Bit != 0
	switch h.SubCPU {
	default:
		h.p.errorf("mach-o error: unknown CPU")
	case machoSubCPU386:
		w.order = binary.LittleEndian
	}

	loadSize := 0
	for _, seg := range h.Segments {
		loadSize += w.segmentSize(seg)
	}
	for _, l := range h.Loads {
		loadSize += 4 * (2 + len(l.Data))
	}

	w.encode(
		w.if64(macho.Magic64, macho.Magic32),
		uint32(h.CPU),
		uint32(h.SubCPU),
		uint32(h.FileType),
		uint32(len(h.Loads)+len(h.Segments)),
		uint32(loadSize),
		uint32(1),
		w.if64(uint32(0), nil),
	)

	for _, seg := range h.Segments {
		w.encode(
			w.if64(uint32(macho.LoadCmdSegment64), uint32(macho.LoadCmdSegment)),
			uint32(w.segmentSize(seg)),
			zeroPad(seg.Name, 16),
			seg.VirtAddr,
			seg.VirtSize,
			seg.FileOffset,
			seg.FileSize,
			seg.Prot1,
			seg.Prot2,
			uint32(len(seg.Sections)),
			seg.Flags,
		)
		for _, sect := range seg.Sections {
			w.encode(
				zeroPad(sect.Name, 16),
				zeroPad(seg.Name, 16),
				sect.Addr,
				sect.Size,
				sect.Offset,
				sect.Align,
				sect.Reloc,
				sect.Nreloc,
				sect.Flags,
				sect.Res1,
				sect.Res2,
				w.if64(uint32(0), nil),
			)
		}
	}

	for _, load := range h.Loads {
		w.encode(
			load.Type,
			uint32(4*(2+len(load.Data))),
			load.Data,
		)
	}

	return w.dst
}
