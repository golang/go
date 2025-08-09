package gosym

import (
	"fmt"
	"iter"
)

// pcValue represents a program counter and its associated value from the pctab.
type pcValue struct {
	// PC is the program counter address.
	PC uint64
	// Val is the associated value (line number, file index, etc.).
	Val int32
}

func (t *Table) firstPcValue(offset uint32) (int32, error) {
	pcValueSeq, err := t.iterPCValues(offset, 0)
	if err != nil {
		return 0, err
	}
	for pcValue := range pcValueSeq {
		return pcValue.Val, nil
	}
	return 0, fmt.Errorf("%w: no pc value found", ErrCorrupted)
}

func (t *Table) lastPc(offset uint32, entryPc uint64) (uint64, error) {
	pcValueSeq, err := t.iterPCValues(offset, entryPc)
	if err != nil {
		return 0, err
	}
	pc := entryPc
	for pcValue := range pcValueSeq {
		pc = pcValue.PC
	}
	return pc, nil
}

func (t *Table) pcValue(offset uint32, entryPc uint64, pc uint64) (int32, error) {
	pcValueSeq, err := t.iterPCValues(offset, entryPc)
	if err != nil {
		return 0, err
	}
	for pcValue := range pcValueSeq {
		if pcValue.PC > pc {
			return pcValue.Val, nil
		}
	}
	return 0, fmt.Errorf("%w: pc %#x value outside of the function", ErrPcNotFound, pc)
}

func (t *Table) iterPCValues(off uint32, entryPc uint64) (iter.Seq[pcValue], error) {
	offset := t.pcTab[0] + uint64(off)
	if offset >= t.pcTab[1] {
		return nil, fmt.Errorf("%w: pctab offset out of bounds", ErrCorrupted)
	}
	pcValueSeq := t.pclntab[offset:t.pcTab[1]]
	val := int32(-1)
	return func(yield func(pcValue) bool) {
		first := true
		for {
			uvdelta, n := decodeVarint(pcValueSeq)
			if !first && uvdelta == 0 {
				return
			}
			first = false
			var vdelta int32
			if (uvdelta & 1) != 0 {
				vdelta = int32(^(uvdelta >> 1))
			} else {
				vdelta = int32(uvdelta >> 1)
			}
			pcValueSeq = pcValueSeq[n:]
			pcdelta, n := decodeVarint(pcValueSeq)
			pcValueSeq = pcValueSeq[n:]
			entryPc += uint64(pcdelta * uint32(t.pcQuantum))
			val += vdelta
			if !yield(pcValue{PC: entryPc, Val: val}) {
				return
			}
		}
	}, nil
}

func decodeVarint(buf []byte) (uint32, int) {
	var result uint32
	var shift uint
	var bytesRead int

	for i, b := range buf {
		if i >= 5 {
			return 0, 0
		}

		result |= uint32(b&0x7F) << shift
		bytesRead++

		if b&0x80 == 0 {
			return result, bytesRead
		}

		shift += 7
	}

	return 0, 0
}
