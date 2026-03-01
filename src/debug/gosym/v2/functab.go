package gosym

import (
	"encoding/binary"
	"fmt"
)

func functabFieldSize(version version, ptrSize uint8) uint64 {
	switch version {
	case ver12, ver116:
		return uint64(ptrSize)
	default:
		return 4
	}
}

func (t *Table) functabCount() uint32 {
	return t.nfunctab
}

func (t *Table) functabField(idx uint32, field uint32) (uint64, error) {
	fieldSize := functabFieldSize(t.version, t.ptrSize)
	offset := t.functab[0] + uint64(2*idx+field)*fieldSize

	if offset+fieldSize > t.functab[1] {
		return 0, fmt.Errorf("%w: function table entry out of bounds", ErrCorrupted)
	}

	var pc uint64
	if fieldSize == 4 {
		pc = uint64(binary.LittleEndian.Uint32(t.pclntab[offset:]))
	} else {
		pc = binary.LittleEndian.Uint64(t.pclntab[offset:])
	}

	return pc, nil
}

func (t *Table) functabPc(idx uint32) (uint64, error) {
	pc, err := t.functabField(idx, 0)
	if err != nil {
		return 0, err
	}
	pc += t.textRange[0]
	return pc, nil
}

func (t *Table) functabOff(idx uint32) (uint64, error) {
	return t.functabField(idx, 1)
}

func (t *Table) functabIdxByPc(pc uint64) (uint32, error) {
	if pc < t.pcRange[0] || pc >= t.pcRange[1] {
		return 0, fmt.Errorf("%w: pc out of global range", ErrPcNotFound)
	}

	lo := uint32(0)
	hi := t.functabCount()
	for hi-lo > 1 {
		mid := (lo + hi) / 2
		midPC, err := t.functabPc(mid)
		if err != nil {
			return 0, err
		}
		if pc < midPC {
			hi = mid
		} else {
			lo = mid
		}
	}

	return lo, nil
}
