// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"debug/macho"
	"encoding/binary"
	"fmt"
	"io/ioutil"
	"strings"
	"testing"
)

// Test macho writing by checking that each generated prog can be written
// and then read back using debug/macho to get the same prog.
// Also check against golden testdata file.
var machoWriteTests = []struct {
	name   string
	golden bool
	prog   *Prog
}{
	// amd64 exit 9
	{
		name:   "exit9",
		golden: true,
		prog: &Prog{
			GOARCH:       "amd64",
			UnmappedSize: 0x1000,
			Entry:        0x1000,
			Segments: []*Segment{
				{
					Name:       "text",
					VirtAddr:   0x1000,
					VirtSize:   13,
					FileOffset: 0,
					FileSize:   13,
					Data: []byte{
						0xb8, 0x01, 0x00, 0x00, 0x02, // MOVL $0x2000001, AX
						0xbf, 0x09, 0x00, 0x00, 0x00, // MOVL $9, DI
						0x0f, 0x05, // SYSCALL
						0xf4, // HLT
					},
					Sections: []*Section{
						{
							Name:     "text",
							VirtAddr: 0x1000,
							Size:     13,
							Align:    64,
						},
					},
				},
			},
		},
	},

	// amd64 write hello world & exit 9
	{
		name:   "hello",
		golden: true,
		prog: &Prog{
			GOARCH:       "amd64",
			UnmappedSize: 0x1000,
			Entry:        0x1000,
			Segments: []*Segment{
				{
					Name:       "text",
					VirtAddr:   0x1000,
					VirtSize:   35,
					FileOffset: 0,
					FileSize:   35,
					Data: []byte{
						0xb8, 0x04, 0x00, 0x00, 0x02, // MOVL $0x2000001, AX
						0xbf, 0x01, 0x00, 0x00, 0x00, // MOVL $1, DI
						0xbe, 0x00, 0x30, 0x00, 0x00, // MOVL $0x3000, SI
						0xba, 0x0c, 0x00, 0x00, 0x00, // MOVL $12, DX
						0x0f, 0x05, // SYSCALL
						0xb8, 0x01, 0x00, 0x00, 0x02, // MOVL $0x2000001, AX
						0xbf, 0x09, 0x00, 0x00, 0x00, // MOVL $9, DI
						0x0f, 0x05, // SYSCALL
						0xf4, // HLT
					},
					Sections: []*Section{
						{
							Name:     "text",
							VirtAddr: 0x1000,
							Size:     35,
							Align:    64,
						},
					},
				},
				{
					Name:       "data",
					VirtAddr:   0x2000,
					VirtSize:   12,
					FileOffset: 0x1000,
					FileSize:   12,
					Data:       []byte("hello world\n"),
					Sections: []*Section{
						{
							Name:     "data",
							VirtAddr: 0x2000,
							Size:     12,
							Align:    64,
						},
					},
				},
			},
		},
	},

	// amd64 write hello world from rodata & exit 0
	{
		name:   "helloro",
		golden: true,
		prog: &Prog{
			GOARCH:       "amd64",
			UnmappedSize: 0x1000,
			Entry:        0x1000,
			Segments: []*Segment{
				{
					Name:       "text",
					VirtAddr:   0x1000,
					VirtSize:   0x100c,
					FileOffset: 0,
					FileSize:   0x100c,
					Data: concat(
						[]byte{
							0xb8, 0x04, 0x00, 0x00, 0x02, // MOVL $0x2000001, AX
							0xbf, 0x01, 0x00, 0x00, 0x00, // MOVL $1, DI
							0xbe, 0x00, 0x30, 0x00, 0x00, // MOVL $0x3000, SI
							0xba, 0x0c, 0x00, 0x00, 0x00, // MOVL $12, DX
							0x0f, 0x05, // SYSCALL
							0xb8, 0x01, 0x00, 0x00, 0x02, // MOVL $0x2000001, AX
							0xbf, 0x00, 0x00, 0x00, 0x00, // MOVL $0, DI
							0x0f, 0x05, // SYSCALL
							0xf4, // HLT
						},
						make([]byte, 0x1000-35),
						[]byte("hello world\n"),
					),
					Sections: []*Section{
						{
							Name:     "text",
							VirtAddr: 0x1000,
							Size:     35,
							Align:    64,
						},
						{
							Name:     "rodata",
							VirtAddr: 0x2000,
							Size:     12,
							Align:    64,
						},
					},
				},
			},
		},
	},
}

func concat(xs ...[]byte) []byte {
	var out []byte
	for _, x := range xs {
		out = append(out, x...)
	}
	return out
}

func TestMachoWrite(t *testing.T) {
	for _, tt := range machoWriteTests {
		name := tt.prog.GOARCH + "." + tt.name
		prog := cloneProg(tt.prog)
		var f machoFormat
		vsize, fsize := f.headerSize(prog)
		shiftProg(prog, vsize, fsize)
		var buf bytes.Buffer
		f.write(&buf, prog)
		if false { // enable to debug
			ioutil.WriteFile("a.out", buf.Bytes(), 0777)
		}
		read, err := machoRead(machoArches[tt.prog.GOARCH], buf.Bytes())
		if err != nil {
			t.Errorf("%s: reading mach-o output:\n\t%v", name, err)
			continue
		}
		diffs := diffProg(read, prog)
		if diffs != nil {
			t.Errorf("%s: mismatched prog:\n\t%s", name, strings.Join(diffs, "\n\t"))
			continue
		}
		if !tt.golden {
			continue
		}
		checkGolden(t, buf.Bytes(), "testdata/macho."+name)
	}
}

// machoRead reads the mach-o file in data and returns a corresponding prog.
func machoRead(arch machoArch, data []byte) (*Prog, error) {
	f, err := macho.NewFile(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}

	var errors []string
	errorf := func(format string, args ...interface{}) {
		errors = append(errors, fmt.Sprintf(format, args...))
	}

	magic := uint32(0xFEEDFACE)
	if arch.CPU&macho64Bit != 0 {
		magic |= 1
	}
	if f.Magic != magic {
		errorf("header: Magic = %#x, want %#x", f.Magic, magic)
	}
	if f.Cpu != macho.CpuAmd64 {
		errorf("header: CPU = %#x, want %#x", f.Cpu, macho.CpuAmd64)
	}
	if f.SubCpu != 3 {
		errorf("header: SubCPU = %#x, want %#x", f.SubCpu, 3)
	}
	if f.Type != 2 {
		errorf("header: FileType = %d, want %d", f.Type, 2)
	}
	if f.Flags != 1 {
		errorf("header: Flags = %d, want %d", f.Flags, 1)
	}

	msects := f.Sections
	var limit uint64
	prog := new(Prog)
	for _, load := range f.Loads {
		switch load := load.(type) {
		default:
			errorf("unexpected macho load %T %x", load, load.Raw())

		case macho.LoadBytes:
			if len(load) < 8 || len(load)%4 != 0 {
				errorf("unexpected load length %d", len(load))
				continue
			}
			cmd := f.ByteOrder.Uint32(load)
			switch macho.LoadCmd(cmd) {
			default:
				errorf("unexpected macho load cmd %s", macho.LoadCmd(cmd))
			case macho.LoadCmdUnixThread:
				data := make([]uint32, len(load[8:])/4)
				binary.Read(bytes.NewReader(load[8:]), f.ByteOrder, data)
				if len(data) != 44 {
					errorf("macho thread len(data) = %d, want 42", len(data))
					continue
				}
				if data[0] != 4 {
					errorf("macho thread type = %d, want 4", data[0])
				}
				if data[1] != uint32(len(data))-2 {
					errorf("macho thread desc len = %d, want %d", data[1], uint32(len(data))-2)
					continue
				}
				for i, val := range data[2:] {
					switch i {
					default:
						if val != 0 {
							errorf("macho thread data[%d] = %#x, want 0", i, val)
						}
					case 32:
						prog.Entry = Addr(val)
					case 33:
						prog.Entry |= Addr(val) << 32
					}
				}
			}

		case *macho.Segment:
			if load.Addr < limit {
				errorf("segments out of order: %q at %#x after %#x", load.Name, load.Addr, limit)
			}
			limit = load.Addr + load.Memsz
			if load.Name == "__PAGEZERO" || load.Addr == 0 && load.Filesz == 0 {
				if load.Name != "__PAGEZERO" {
					errorf("segment with Addr=0, Filesz=0 is named %q, want %q", load.Name, "__PAGEZERO")
				} else if load.Addr != 0 || load.Filesz != 0 {
					errorf("segment %q has Addr=%#x, Filesz=%d, want Addr=%#x, Filesz=%d", load.Name, load.Addr, load.Filesz, 0, 0)
				}
				prog.UnmappedSize = Addr(load.Memsz)
				continue
			}

			if !strings.HasPrefix(load.Name, "__") {
				errorf("segment name %q does not begin with %q", load.Name, "__")
			}
			if strings.ToUpper(load.Name) != load.Name {
				errorf("segment name %q is not all upper case", load.Name)
			}

			seg := &Segment{
				Name:       strings.ToLower(strings.TrimPrefix(load.Name, "__")),
				VirtAddr:   Addr(load.Addr),
				VirtSize:   Addr(load.Memsz),
				FileOffset: Addr(load.Offset),
				FileSize:   Addr(load.Filesz),
			}
			prog.Segments = append(prog.Segments, seg)

			data, err := load.Data()
			if err != nil {
				errorf("loading data from %q: %v", load.Name, err)
			}
			seg.Data = data

			var maxprot, prot uint32
			if load.Name == "__TEXT" {
				maxprot, prot = 7, 5
			} else {
				maxprot, prot = 3, 3
			}
			if load.Maxprot != maxprot || load.Prot != prot {
				errorf("segment %q protection is %d, %d, want %d, %d",
					load.Maxprot, load.Prot, maxprot, prot)
			}

			for len(msects) > 0 && msects[0].Addr < load.Addr+load.Memsz {
				msect := msects[0]
				msects = msects[1:]

				if msect.Offset > 0 && prog.HeaderSize == 0 {
					prog.HeaderSize = Addr(msect.Offset)
					if seg.FileOffset != 0 {
						errorf("initial segment %q does not map header", load.Name)
					}
					seg.VirtAddr += prog.HeaderSize
					seg.VirtSize -= prog.HeaderSize
					seg.FileOffset += prog.HeaderSize
					seg.FileSize -= prog.HeaderSize
					seg.Data = seg.Data[prog.HeaderSize:]
				}

				if msect.Addr < load.Addr {
					errorf("section %q at address %#x is missing segment", msect.Name, msect.Addr)
					continue
				}

				if !strings.HasPrefix(msect.Name, "__") {
					errorf("section name %q does not begin with %q", msect.Name, "__")
				}
				if strings.ToLower(msect.Name) != msect.Name {
					errorf("section name %q is not all lower case", msect.Name)
				}
				if msect.Seg != load.Name {
					errorf("section %q is lists segment name %q, want %q",
						msect.Name, msect.Seg, load.Name)
				}
				if uint64(msect.Offset) != uint64(load.Offset)+msect.Addr-load.Addr {
					errorf("section %q file offset is %#x, want %#x",
						msect.Name, msect.Offset, load.Offset+msect.Addr-load.Addr)
				}
				if msect.Reloff != 0 || msect.Nreloc != 0 {
					errorf("section %q has reloff %d,%d, want %d,%d",
						msect.Name, msect.Reloff, msect.Nreloc, 0, 0)
				}
				flags := uint32(0)
				if msect.Name == "__text" {
					flags = 0x400
				}
				if msect.Offset == 0 {
					flags = 1
				}
				if msect.Flags != flags {
					errorf("section %q flags = %#x, want %#x", msect.Flags, flags)
				}
				sect := &Section{
					Name:     strings.ToLower(strings.TrimPrefix(msect.Name, "__")),
					VirtAddr: Addr(msect.Addr),
					Size:     Addr(msect.Size),
					Align:    1 << msect.Align,
				}
				seg.Sections = append(seg.Sections, sect)
			}
		}
	}

	for _, msect := range msects {
		errorf("section %q has no segment", msect.Name)
	}

	limit = 0
	for _, msect := range f.Sections {
		if msect.Addr < limit {
			errorf("sections out of order: %q at %#x after %#x", msect.Name, msect.Addr, limit)
		}
		limit = msect.Addr + msect.Size
	}

	err = nil
	if errors != nil {
		err = fmt.Errorf("%s", strings.Join(errors, "\n\t"))
	}
	return prog, err
}
