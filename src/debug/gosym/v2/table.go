package gosym

import (
	"bytes"
	"encoding/binary"
	"fmt"
)

// version represents the supported pclntab versions.
type version int

// pclnTabVersion constants
const (
	ver12 version = iota
	ver116
	ver118
	ver120
)

const (
	go12Magic  = 0xfffffffb
	go116Magic = 0xfffffffa
	go118Magic = 0xfffffff0
	go120Magic = 0xfffffff1
)

type metadata struct {
	version       version
	wrapperFuncID uint8

	// Quantum for pc values iteration
	pcQuantum uint8
	// The pointer size
	ptrSize uint8
	// The number of function entries
	nfunctab uint32
	// The number of file entries
	nfiletab uint32

	// For ver118/120, the text start address (used to relocate PCs); otherwise zeros
	textRange [2]uint64
	// Pc range
	pcRange [2]uint64

	// Data offsets within the pclntab
	// The offset of the file table
	filetab [2]uint64
	// The offset of the function table
	functab [2]uint64
	// The blob of function metadata
	funcdata [2]uint64
	// The function name table
	funcnametab [2]uint64
	// The compile unit table
	cutab [2]uint64
	// The pc table
	pcTab [2]uint64
}

func parseObject(obj Object) (*Table, error) {
	var (
		pclntab       SectionHeader
		pclntabData   []byte
		noptrdataData []byte
		rodata        []SectionHeader
		rodataData    [][]byte
		text          SectionHeader
	)

	headers, err := obj.Sections()
	if err != nil {
		return nil, err
	}
	for i, h := range headers {
		switch h.Name {
		case ".gopclntab", "__gopclntab":
			pclntab = h
			pclntabData, err = obj.SectionData(int8(i))
			if err != nil {
				return nil, err
			}
		case ".noptrdata", "__noptrdata":
			noptrdataData, err = obj.SectionData(int8(i))
			if err != nil {
				return nil, err
			}
		case ".rodata", "__rodata":
			rodata = append(rodata, h)
			data, err := obj.SectionData(int8(i))
			if err != nil {
				return nil, err
			}
			rodataData = append(rodataData, data)
		case ".text", "__text":
			text = h
		}
	}

	// Search for the moduledata structure
	addrBytes := make([]byte, 8)
	obj.Endian().PutUint64(addrBytes, pclntab.Addr)

	offsets := findAll(noptrdataData, addrBytes)
	var moduledata moduledata
	for _, offset := range offsets {
		var valid bool
		moduledata, valid = tryParseModuleDataAt(
			text, rodata,
			noptrdataData, rodataData,
			offset,
		)
		if valid {
			goto found
		}
	}
	return nil, fmt.Errorf("%w: moduledata not found", ErrCorrupted)

found:
	return parseTable(pclntabData, moduledata.gofunc, moduledata.textRange, moduledata.pcRange)
}

type moduledata struct {
	textRange, pcRange [2]uint64
	gofunc             []byte
}

// Module data offsets
const (
	moduledataMinPCOffset       = 160
	moduledataTextOffset        = moduledataMinPCOffset + 16
	moduledataBssOffset         = moduledataTextOffset + 48
	moduledataGcdataOffset      = moduledataBssOffset + 56
	moduledataTypesOffset       = moduledataGcdataOffset + 16
	moduledataGofuncOffset      = moduledataTypesOffset + 24
	moduledataTextsectMapOffset = moduledataTypesOffset + 32
)

func tryParseModuleDataAt(
	textHeader SectionHeader,
	rodataHeaders []SectionHeader,
	noptrdataData []byte,
	rodataData [][]byte,
	offset int,
) (moduledata, bool) {
	// Parse text range
	textStart := offset + moduledataTextOffset
	if textStart+16 > len(noptrdataData) {
		return moduledata{}, false
	}
	text := binary.LittleEndian.Uint64(noptrdataData[textStart:])
	etext := binary.LittleEndian.Uint64(noptrdataData[textStart+8:])
	if text > etext || text < textHeader.Addr || etext > textHeader.Addr+textHeader.Size {
		return moduledata{}, false
	}

	// Parse types range
	typesStart := offset + moduledataTypesOffset
	if typesStart+16 > len(noptrdataData) {
		return moduledata{}, false
	}
	types := binary.LittleEndian.Uint64(noptrdataData[typesStart:])
	etypes := binary.LittleEndian.Uint64(noptrdataData[typesStart+8:])
	if types > etypes {
		return moduledata{}, false
	}
	var valid bool
	for _, h := range rodataHeaders {
		if h.Addr <= types && etypes <= h.Addr+h.Size {
			valid = true
			break
		}
	}
	if !valid {
		return moduledata{}, false
	}

	// Parse textsect map
	textsectMapOffset := offset + moduledataTextsectMapOffset
	if textsectMapOffset+16 > len(noptrdataData) {
		return moduledata{}, false
	}
	textsectMapPtr := binary.LittleEndian.Uint64(noptrdataData[textsectMapOffset:])
	textsectMapLen := binary.LittleEndian.Uint64(noptrdataData[textsectMapOffset+8:])

	valid = false
	for _, h := range rodataHeaders {
		if textsectMapPtr < h.Addr {
			continue
		}
		textsectMapDataOffset := int(textsectMapPtr - h.Addr)
		textsectMapDataLen := int(textsectMapLen * 24)
		if textsectMapDataOffset+textsectMapDataLen < int(h.Size) {
			valid = true
			break
		}
	}
	if !valid {
		return moduledata{}, false
	}

	// Parse BSS range
	bssOffset := offset + moduledataBssOffset
	if bssOffset+16 > len(noptrdataData) {
		return moduledata{}, false
	}

	// Parse gofunc offset
	gofuncOffset := offset + moduledataGofuncOffset
	if gofuncOffset+8 > len(noptrdataData) {
		return moduledata{}, false
	}
	gofunc := binary.LittleEndian.Uint64(noptrdataData[gofuncOffset:])

	// Parse gcdata offset
	gcdataOffset := offset + moduledataGcdataOffset
	if gcdataOffset+8 > len(noptrdataData) {
		return moduledata{}, false
	}
	gcdata := binary.LittleEndian.Uint64(noptrdataData[gcdataOffset:])

	rodataIdx := -1
	for i, h := range rodataHeaders {
		if h.Addr <= gofunc && gcdata <= h.Addr+h.Size {
			rodataIdx = i
			gofunc -= h.Addr
			gcdata -= h.Addr
			break
		}
	}
	if rodataIdx == -1 {
		return moduledata{}, false
	}

	// Parse min/max PC
	minPCOffset := offset + moduledataMinPCOffset
	if minPCOffset+16 > len(noptrdataData) {
		return moduledata{}, false
	}
	minPC := binary.LittleEndian.Uint64(noptrdataData[minPCOffset:])
	maxPC := binary.LittleEndian.Uint64(noptrdataData[minPCOffset+8:])

	return moduledata{
		textRange: [2]uint64{text, etext},
		pcRange:   [2]uint64{minPC, maxPC},
		gofunc:    rodataData[rodataIdx][gofunc:gcdata],
	}, true
}

func findAll(haystack, needle []byte) []int {
	var offsets []int
	start := 0
	for {
		idx := bytes.Index(haystack[start:], needle)
		if idx == -1 {
			break
		}
		offsets = append(offsets, start+idx)
		start += idx + 1
	}
	return offsets
}

func parseTable(
	pclntab, gofunc []byte,
	textRange, pcRange [2]uint64,
) (*Table, error) {
	magic := binary.LittleEndian.Uint32(pclntab[0:4])
	var version version
	switch magic {
	case go12Magic:
		version = ver12
	case go116Magic:
		version = ver116
	case go118Magic:
		version = ver118
	case go120Magic:
		version = ver120
	default:
		return nil, fmt.Errorf("%w: unsupported pclntab magic: %x", ErrCorrupted, magic)
	}

	if pclntab[4] != 0 || pclntab[5] != 0 {
		return nil, fmt.Errorf("%w: unexpected pclntab non-zero header padding", ErrCorrupted)
	}

	quantum := uint8(pclntab[6])
	ptrSize := uint8(pclntab[7])

	if ptrSize != 4 && ptrSize != 8 {
		return nil, fmt.Errorf("%w: invalid pointer size in pclntab: %d", ErrCorrupted, ptrSize)
	}

	t := &Table{
		metadata: metadata{
			version:   version,
			pcQuantum: quantum,
			ptrSize:   ptrSize,
			textRange: textRange,
			pcRange:   pcRange,
		},
		pclntab: pclntab,
		gofunc:  gofunc,
	}

	switch version {
	case ver118, ver120:
		err := parseMetadata118(t)
		if err != nil {
			return nil, err
		}
	case ver116:
		err := parseMetadata116(t)
		if err != nil {
			return nil, err
		}
	case ver12:
		err := parseMetadata12(t)
		if err != nil {
			return nil, err
		}
	}

	// Now we need to figure out the wrapper function ID. The value is not
	// stable across versions, and it's not always easy to get the go version.
	// Instead we look for known wrapper function.
	wrapperFuncID := uint8(255)
	for f := range t.Functions() {
		name, err := f.Name()
		if err == nil && name.Value() == "runtime.deferreturn" {
			wrapperFuncID, err = t.funcdataFuncID(f.offset)
			if err != nil {
				break
			}
		}
	}
	t.wrapperFuncID = wrapperFuncID

	return t, nil
}

func parseMetadata118(t *Table) error {
	pclntablen := uint64(len(t.pclntab))
	readWord := func(offset uint32) (uint64, error) {
		start := 8 + int(offset)*int(t.ptrSize)
		if start+int(t.ptrSize) > int(pclntablen) {
			return 0, fmt.Errorf("%w: pclntab too short for word at offset: %d", ErrCorrupted, offset)
		}
		if t.ptrSize == 8 {
			return binary.LittleEndian.Uint64(t.pclntab[start : start+8]), nil
		}
		return uint64(binary.LittleEndian.Uint32(t.pclntab[start : start+4])), nil
	}

	nfunctab, err := readWord(0)
	if err != nil {
		return err
	}
	nfiletab, err := readWord(1)
	if err != nil {
		return err
	}
	funcnametab, err := readWord(3)
	if err != nil {
		return err
	}
	if funcnametab >= pclntablen {
		return fmt.Errorf("%w: funcnametab out of bounds", ErrCorrupted)
	}
	cutab, err := readWord(4)
	if err != nil {
		return err
	}
	if cutab >= pclntablen {
		return fmt.Errorf("%w: cutab out of bounds", ErrCorrupted)
	}
	filetab, err := readWord(5)
	if err != nil {
		return err
	}
	if filetab >= pclntablen {
		return fmt.Errorf("%w: filetab out of bounds", ErrCorrupted)
	}
	pctab, err := readWord(6)
	if err != nil {
		return err
	}
	if pctab >= pclntablen {
		return fmt.Errorf("%w: pctab out of bounds", ErrCorrupted)
	}
	functab, err := readWord(7)
	if err != nil {
		return err
	}
	functablen := (nfunctab*2 + 1) * functabFieldSize(t.version, t.ptrSize)
	if functab+functablen > pclntablen {
		return fmt.Errorf("%w: functab out of bounds", ErrCorrupted)
	}
	t.nfunctab = uint32(nfunctab)
	t.nfiletab = uint32(nfiletab)
	t.filetab = [2]uint64{filetab, pclntablen}
	t.functab = [2]uint64{functab, functab + functablen}
	t.funcdata = [2]uint64{functab, pclntablen}
	t.funcnametab = [2]uint64{funcnametab, pclntablen}
	t.cutab = [2]uint64{cutab, pclntablen}
	t.pcTab = [2]uint64{pctab, pclntablen}
	return nil
}

func parseMetadata116(t *Table) error {
	pclntablen := uint64(len(t.pclntab))
	readWord := func(offset uint32) (uint64, error) {
		start := 8 + int(offset)*int(t.ptrSize)
		if start+int(t.ptrSize) > int(pclntablen) {
			return 0, fmt.Errorf("%w: pclntab too short for word at offset: %d", ErrCorrupted, offset)
		}
		if t.ptrSize == 8 {
			return binary.LittleEndian.Uint64(t.pclntab[start : start+8]), nil
		}
		return uint64(binary.LittleEndian.Uint32(t.pclntab[start : start+4])), nil
	}

	nfunctab, err := readWord(0)
	if err != nil {
		return err
	}
	nfiletab, err := readWord(1)
	if err != nil {
		return err
	}
	funcnametab, err := readWord(2)
	if err != nil {
		return err
	}
	if funcnametab >= pclntablen {
		return fmt.Errorf("%w: funcnametab out of bounds", ErrCorrupted)
	}
	cutab, err := readWord(3)
	if err != nil {
		return err
	}
	if cutab >= pclntablen {
		return fmt.Errorf("%w: cutab out of bounds", ErrCorrupted)
	}
	filetab, err := readWord(4)
	if err != nil {
		return err
	}
	if filetab >= pclntablen {
		return fmt.Errorf("%w: filetab out of bounds", ErrCorrupted)
	}
	pctab, err := readWord(5)
	if err != nil {
		return err
	}
	if pctab >= pclntablen {
		return fmt.Errorf("%w: pctab out of bounds", ErrCorrupted)
	}
	functab, err := readWord(6)
	if err != nil {
		return err
	}
	functablen := (nfunctab*2 + 1) * functabFieldSize(t.version, t.ptrSize)
	if functab+functablen > pclntablen {
		return fmt.Errorf("%w: functab out of bounds", ErrCorrupted)
	}
	t.nfunctab = uint32(nfunctab)
	t.nfiletab = uint32(nfiletab)
	t.filetab = [2]uint64{filetab, pclntablen}
	t.functab = [2]uint64{functab, functab + functablen}
	t.funcdata = [2]uint64{functab, pclntablen}
	t.funcnametab = [2]uint64{funcnametab, pclntablen}
	t.cutab = [2]uint64{cutab, pclntablen}
	t.pcTab = [2]uint64{pctab, pclntablen}
	return nil
}

func parseMetadata12(t *Table) error {
	pclntablen := uint64(len(t.pclntab))
	readWord := func(offset uint32) (uint64, error) {
		start := 8 + int(offset)*int(t.ptrSize)
		if start+int(t.ptrSize) > int(pclntablen) {
			return 0, fmt.Errorf("%w: pclntab too short for word at offset: %d", ErrCorrupted, offset)
		}
		if t.ptrSize == 8 {
			return binary.LittleEndian.Uint64(t.pclntab[start : start+8]), nil
		}
		return uint64(binary.LittleEndian.Uint32(t.pclntab[start : start+4])), nil
	}

	nfunctab, err := readWord(0)
	if err != nil {
		return err
	}
	functab := uint64(8 + t.ptrSize)
	functablen := (nfunctab*2 + 1) * functabFieldSize(t.version, t.ptrSize)
	if functab+functablen > pclntablen {
		return fmt.Errorf("%w: functab out of bounds", ErrCorrupted)
	}
	readUint32 := func(offset uint32) (uint32, error) {
		if uint64(offset+4) > pclntablen {
			return 0, fmt.Errorf("%w: pclntab too short for uint32 at offset: %d", ErrCorrupted, offset)
		}
		return binary.LittleEndian.Uint32(t.pclntab[offset : offset+4]), nil
	}
	filetab, err := readUint32(uint32(functab + functablen))
	if err != nil {
		return err
	}
	nfiletab, err := readUint32(filetab)
	if err != nil {
		return err
	}
	t.nfunctab = uint32(nfunctab)
	t.nfiletab = uint32(nfiletab)
	t.filetab = [2]uint64{uint64(filetab), uint64(filetab + nfiletab*4)}
	t.functab = [2]uint64{functab, functab + functablen}
	t.funcdata = [2]uint64{0, pclntablen}
	t.funcnametab = [2]uint64{0, pclntablen}
	t.cutab = [2]uint64{0, 0}
	t.pcTab = [2]uint64{0, pclntablen}
	return nil
}
