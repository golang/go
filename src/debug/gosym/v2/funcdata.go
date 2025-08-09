package gosym

import (
	"encoding/binary"
	"fmt"
	"math"
)

func funcdataEntryOffsetSize(version version, ptrSize uint8) uint64 {
	switch version {
	case ver12, ver116:
		return uint64(ptrSize)
	default:
		return 4
	}
}

func (t *Table) funcdataFieldOffset(field uint64) uint64 {
	return funcdataEntryOffsetSize(t.version, t.ptrSize) + (field-1)*4
}

func (t *Table) funcdataField(funcOff uint64, field uint64) (uint32, error) {
	offset := t.funcdata[0] + funcOff + t.funcdataFieldOffset(field)
	if offset+4 > t.funcdata[1] {
		return 0, fmt.Errorf("%w: function data out of bounds: off=%d field=%d", ErrCorrupted, offset, field)
	}
	return binary.LittleEndian.Uint32(t.pclntab[offset:]), nil
}

func (t *Table) funcdataNameOff(funcOff uint64) (uint32, error) {
	return t.funcdataField(funcOff, 1)
}

func (t *Table) funcdataDeferReturn(funcOff uint64) (uint32, error) {
	return t.funcdataField(funcOff, 3)
}

func (t *Table) funcdataPcfile(funcOff uint64) (uint32, error) {
	return t.funcdataField(funcOff, 5)
}

func (t *Table) funcdataPcln(funcOff uint64) (uint32, error) {
	return t.funcdataField(funcOff, 6)
}

func (t *Table) funcdataNpcdata(funcOff uint64) (uint32, error) {
	return t.funcdataField(funcOff, 7)
}

func (t *Table) funcdataCuOffset(funcOff uint64) (uint32, error) {
	return t.funcdataField(funcOff, 8)
}

func (t *Table) funcdataStartLine(funcOff uint64) (uint32, error) {
	line, err := t.funcdataField(funcOff, 9)
	if err != nil {
		return 0, err
	}
	return line, nil
}

func (t *Table) funcdataFuncID(funcOff uint64) (uint8, error) {
	funcIDOff := t.funcdata[0] + funcOff + t.funcdataFieldOffset(10)
	if funcIDOff >= t.funcdata[1] {
		return 0, fmt.Errorf("%w: function data out of bounds, off=%d", ErrCorrupted, funcIDOff)
	}
	return t.pclntab[funcIDOff], nil
}

func (t *Table) funcdataFileNoToOff(funcOff uint64, fno uint32) (uint32, error) {
	if t.version == ver12 {
		return uint32(fno * 4), nil
	}
	cuOff, err := t.funcdataCuOffset(funcOff)
	if err != nil {
		return 0, err
	}
	offOff := t.cutab[0] + uint64(cuOff)*4 + uint64(fno)*4
	if offOff+4 > t.cutab[1] {
		return 0, fmt.Errorf("%w: file offset out of bounds", ErrCorrupted)
	}
	off := binary.LittleEndian.Uint32(t.pclntab[offOff:])
	if off == math.MaxUint32 {
		// Valid for non-function entries. We skip them at higher levels,
		// so here we can return an error.
		return 0, fmt.Errorf("%w: no file entry", ErrCorrupted)
	}
	return off, nil
}

func (t *Table) funcdataFileNo(funcOff uint64) (uint32, error) {
	pcfile, err := t.funcdataPcfile(funcOff)
	if err != nil {
		return 0, err
	}
	fno, err := t.firstPcValue(pcfile)
	if err != nil {
		return 0, err
	}
	return uint32(fno), nil
}

func (t *Table) funcdataFileOff(funcOff uint64) (uint32, error) {
	fno, err := t.funcdataFileNo(funcOff)
	if err != nil {
		return 0, err
	}
	return t.funcdataFileNoToOff(funcOff, uint32(fno))
}

func (t *Table) funcdataInlTreeIndex(funcOff uint64) (uint32, error) {
	npcdata, err := t.funcdataNpcdata(funcOff)
	if err != nil {
		return 0, err
	}
	const pcdataInlTreeIndex = 2
	if pcdataInlTreeIndex >= npcdata {
		return 0, fmt.Errorf("%w: inl tree index out of bounds, off=%d", ErrCorrupted, funcOff)
	}
	return t.funcdataField(funcOff, uint64(11+pcdataInlTreeIndex))
}

func (t *Table) funcdataInlTree(funcOff uint64) ([]byte, error) {
	npcdata, err := t.funcdataNpcdata(funcOff)
	if err != nil {
		return nil, err
	}
	nfuncdataOff := t.funcdata[0] + funcOff + t.funcdataFieldOffset(11)
	if nfuncdataOff >= t.funcdata[1] {
		return nil, fmt.Errorf("%w: function data out of bounds, off=%d", ErrCorrupted, nfuncdataOff)
	}
	nfuncdata := t.pclntab[nfuncdataOff]
	const funcdataInlTree = 3
	if funcdataInlTree >= nfuncdata {
		return nil, fmt.Errorf("%w: inl tree out of bounds, off=%d", ErrCorrupted, funcOff)
	}
	inlTreeOff, err := t.funcdataField(funcOff, uint64(11+npcdata+funcdataInlTree))
	if err != nil {
		return nil, err
	}
	if inlTreeOff == math.MaxUint32 {
		// Valid case - this function has no inline functions
		return nil, nil
	}
	return t.gofunc[inlTreeOff:], nil
}
