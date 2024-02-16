// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package decodecounter

import (
	"encoding/binary"
	"fmt"
	"internal/coverage"
	"internal/coverage/slicereader"
	"internal/coverage/stringtab"
	"io"
	"os"
	"strconv"
	"unsafe"
)

// This file contains helpers for reading counter data files created
// during the executions of a coverage-instrumented binary.

type CounterDataReader struct {
	stab     *stringtab.Reader
	args     map[string]string
	osargs   []string
	goarch   string // GOARCH setting from run that produced counter data
	goos     string // GOOS setting from run that produced counter data
	mr       io.ReadSeeker
	hdr      coverage.CounterFileHeader
	ftr      coverage.CounterFileFooter
	shdr     coverage.CounterSegmentHeader
	u32b     []byte
	u8b      []byte
	fcnCount uint32
	segCount uint32
	debug    bool
}

func NewCounterDataReader(fn string, rs io.ReadSeeker) (*CounterDataReader, error) {
	cdr := &CounterDataReader{
		mr:   rs,
		u32b: make([]byte, 4),
		u8b:  make([]byte, 1),
	}
	// Read header
	if err := binary.Read(rs, binary.LittleEndian, &cdr.hdr); err != nil {
		return nil, err
	}
	if cdr.debug {
		fmt.Fprintf(os.Stderr, "=-= counter file header: %+v\n", cdr.hdr)
	}
	if !checkMagic(cdr.hdr.Magic) {
		return nil, fmt.Errorf("invalid magic string: not a counter data file")
	}
	if cdr.hdr.Version > coverage.CounterFileVersion {
		return nil, fmt.Errorf("version data incompatibility: reader is %d data is %d", coverage.CounterFileVersion, cdr.hdr.Version)
	}

	// Read footer.
	if err := cdr.readFooter(); err != nil {
		return nil, err
	}
	// Seek back to just past the file header.
	hsz := int64(unsafe.Sizeof(cdr.hdr))
	if _, err := cdr.mr.Seek(hsz, io.SeekStart); err != nil {
		return nil, err
	}
	// Read preamble for first segment.
	if err := cdr.readSegmentPreamble(); err != nil {
		return nil, err
	}
	return cdr, nil
}

func checkMagic(v [4]byte) bool {
	g := coverage.CovCounterMagic
	return v[0] == g[0] && v[1] == g[1] && v[2] == g[2] && v[3] == g[3]
}

func (cdr *CounterDataReader) readFooter() error {
	ftrSize := int64(unsafe.Sizeof(cdr.ftr))
	if _, err := cdr.mr.Seek(-ftrSize, io.SeekEnd); err != nil {
		return err
	}
	if err := binary.Read(cdr.mr, binary.LittleEndian, &cdr.ftr); err != nil {
		return err
	}
	if !checkMagic(cdr.ftr.Magic) {
		return fmt.Errorf("invalid magic string (not a counter data file)")
	}
	if cdr.ftr.NumSegments == 0 {
		return fmt.Errorf("invalid counter data file (no segments)")
	}
	return nil
}

// readSegmentPreamble reads and consumes the segment header, segment string
// table, and segment args table.
func (cdr *CounterDataReader) readSegmentPreamble() error {
	// Read segment header.
	if err := binary.Read(cdr.mr, binary.LittleEndian, &cdr.shdr); err != nil {
		return err
	}
	if cdr.debug {
		fmt.Fprintf(os.Stderr, "=-= read counter segment header: %+v", cdr.shdr)
		fmt.Fprintf(os.Stderr, " FcnEntries=0x%x StrTabLen=0x%x ArgsLen=0x%x\n",
			cdr.shdr.FcnEntries, cdr.shdr.StrTabLen, cdr.shdr.ArgsLen)
	}

	// Read string table and args.
	if err := cdr.readStringTable(); err != nil {
		return err
	}
	if err := cdr.readArgs(); err != nil {
		return err
	}
	// Seek past any padding to bring us up to a 4-byte boundary.
	if of, err := cdr.mr.Seek(0, io.SeekCurrent); err != nil {
		return err
	} else {
		rem := of % 4
		if rem != 0 {
			pad := 4 - rem
			if _, err := cdr.mr.Seek(pad, io.SeekCurrent); err != nil {
				return err
			}
		}
	}
	return nil
}

func (cdr *CounterDataReader) readStringTable() error {
	b := make([]byte, cdr.shdr.StrTabLen)
	nr, err := cdr.mr.Read(b)
	if err != nil {
		return err
	}
	if nr != int(cdr.shdr.StrTabLen) {
		return fmt.Errorf("error: short read on string table")
	}
	slr := slicereader.NewReader(b, false /* not readonly */)
	cdr.stab = stringtab.NewReader(slr)
	cdr.stab.Read()
	return nil
}

func (cdr *CounterDataReader) readArgs() error {
	b := make([]byte, cdr.shdr.ArgsLen)
	nr, err := cdr.mr.Read(b)
	if err != nil {
		return err
	}
	if nr != int(cdr.shdr.ArgsLen) {
		return fmt.Errorf("error: short read on args table")
	}
	slr := slicereader.NewReader(b, false /* not readonly */)
	sget := func() (string, error) {
		kidx := slr.ReadULEB128()
		if int(kidx) >= cdr.stab.Entries() {
			return "", fmt.Errorf("malformed string table ref")
		}
		return cdr.stab.Get(uint32(kidx)), nil
	}
	nents := slr.ReadULEB128()
	cdr.args = make(map[string]string, int(nents))
	for i := uint64(0); i < nents; i++ {
		k, errk := sget()
		if errk != nil {
			return errk
		}
		v, errv := sget()
		if errv != nil {
			return errv
		}
		if _, ok := cdr.args[k]; ok {
			return fmt.Errorf("malformed args table")
		}
		cdr.args[k] = v
	}
	if argcs, ok := cdr.args["argc"]; ok {
		argc, err := strconv.Atoi(argcs)
		if err != nil {
			return fmt.Errorf("malformed argc in counter data file args section")
		}
		cdr.osargs = make([]string, 0, argc)
		for i := 0; i < argc; i++ {
			arg := cdr.args[fmt.Sprintf("argv%d", i)]
			cdr.osargs = append(cdr.osargs, arg)
		}
	}
	if goos, ok := cdr.args["GOOS"]; ok {
		cdr.goos = goos
	}
	if goarch, ok := cdr.args["GOARCH"]; ok {
		cdr.goarch = goarch
	}
	return nil
}

// OsArgs returns the program arguments (saved from os.Args during
// the run of the instrumented binary) read from the counter
// data file. Not all coverage data files will have os.Args values;
// for example, if a data file is produced by merging coverage
// data from two distinct runs, no os args will be available (an
// empty list is returned).
func (cdr *CounterDataReader) OsArgs() []string {
	return cdr.osargs
}

// Goos returns the GOOS setting in effect for the "-cover" binary
// that produced this counter data file. The GOOS value may be
// empty in the case where the counter data file was produced
// from a merge in which more than one GOOS value was present.
func (cdr *CounterDataReader) Goos() string {
	return cdr.goos
}

// Goarch returns the GOARCH setting in effect for the "-cover" binary
// that produced this counter data file. The GOARCH value may be
// empty in the case where the counter data file was produced
// from a merge in which more than one GOARCH value was present.
func (cdr *CounterDataReader) Goarch() string {
	return cdr.goarch
}

// FuncPayload encapsulates the counter data payload for a single
// function as read from a counter data file.
type FuncPayload struct {
	PkgIdx   uint32
	FuncIdx  uint32
	Counters []uint32
}

// NumSegments returns the number of execution segments in the file.
func (cdr *CounterDataReader) NumSegments() uint32 {
	return cdr.ftr.NumSegments
}

// BeginNextSegment sets up the reader to read the next segment,
// returning TRUE if we do have another segment to read, or FALSE
// if we're done with all the segments (also an error if
// something went wrong).
func (cdr *CounterDataReader) BeginNextSegment() (bool, error) {
	if cdr.segCount >= cdr.ftr.NumSegments {
		return false, nil
	}
	cdr.segCount++
	cdr.fcnCount = 0
	// Seek past footer from last segment.
	ftrSize := int64(unsafe.Sizeof(cdr.ftr))
	if _, err := cdr.mr.Seek(ftrSize, io.SeekCurrent); err != nil {
		return false, err
	}
	// Read preamble for this segment.
	if err := cdr.readSegmentPreamble(); err != nil {
		return false, err
	}
	return true, nil
}

// NumFunctionsInSegment returns the number of live functions
// in the currently selected segment.
func (cdr *CounterDataReader) NumFunctionsInSegment() uint32 {
	return uint32(cdr.shdr.FcnEntries)
}

const supportDeadFunctionsInCounterData = false

// NextFunc reads data for the next function in this current segment
// into "p", returning TRUE if the read was successful or FALSE
// if we've read all the functions already (also an error if
// something went wrong with the read or we hit a premature
// EOF).
func (cdr *CounterDataReader) NextFunc(p *FuncPayload) (bool, error) {
	if cdr.fcnCount >= uint32(cdr.shdr.FcnEntries) {
		return false, nil
	}
	cdr.fcnCount++
	var rdu32 func() (uint32, error)
	if cdr.hdr.CFlavor == coverage.CtrULeb128 {
		rdu32 = func() (uint32, error) {
			var shift uint
			var value uint64
			for {
				_, err := cdr.mr.Read(cdr.u8b)
				if err != nil {
					return 0, err
				}
				b := cdr.u8b[0]
				value |= (uint64(b&0x7F) << shift)
				if b&0x80 == 0 {
					break
				}
				shift += 7
			}
			return uint32(value), nil
		}
	} else if cdr.hdr.CFlavor == coverage.CtrRaw {
		if cdr.hdr.BigEndian {
			rdu32 = func() (uint32, error) {
				n, err := cdr.mr.Read(cdr.u32b)
				if err != nil {
					return 0, err
				}
				if n != 4 {
					return 0, io.EOF
				}
				return binary.BigEndian.Uint32(cdr.u32b), nil
			}
		} else {
			rdu32 = func() (uint32, error) {
				n, err := cdr.mr.Read(cdr.u32b)
				if err != nil {
					return 0, err
				}
				if n != 4 {
					return 0, io.EOF
				}
				return binary.LittleEndian.Uint32(cdr.u32b), nil
			}
		}
	} else {
		panic("internal error: unknown counter flavor")
	}

	// Alternative/experimental path: one way we could handling writing
	// out counter data would be to just memcpy the counter segment
	// out to a file, meaning that a region in the counter memory
	// corresponding to a dead (never-executed) function would just be
	// zeroes. The code path below handles this case.
	var nc uint32
	var err error
	if supportDeadFunctionsInCounterData {
		for {
			nc, err = rdu32()
			if err == io.EOF {
				return false, io.EOF
			} else if err != nil {
				break
			}
			if nc != 0 {
				break
			}
		}
	} else {
		nc, err = rdu32()
	}
	if err != nil {
		return false, err
	}

	// Read package and func indices.
	p.PkgIdx, err = rdu32()
	if err != nil {
		return false, err
	}
	p.FuncIdx, err = rdu32()
	if err != nil {
		return false, err
	}
	if cap(p.Counters) < 1024 {
		p.Counters = make([]uint32, 0, 1024)
	}
	p.Counters = p.Counters[:0]
	for i := uint32(0); i < nc; i++ {
		v, err := rdu32()
		if err != nil {
			return false, err
		}
		p.Counters = append(p.Counters, v)
	}
	return true, nil
}
