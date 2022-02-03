// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fuzz

import (
	"encoding/binary"
	"fmt"
	"math"
	"reflect"
	"unsafe"
)

type mutator struct {
	r       mutatorRand
	scratch []byte // scratch slice to avoid additional allocations
}

func newMutator() *mutator {
	return &mutator{r: newPcgRand()}
}

func (m *mutator) rand(n int) int {
	return m.r.intn(n)
}

func (m *mutator) randByteOrder() binary.ByteOrder {
	if m.r.bool() {
		return binary.LittleEndian
	}
	return binary.BigEndian
}

// chooseLen chooses length of range mutation in range [1,n]. It gives
// preference to shorter ranges.
func (m *mutator) chooseLen(n int) int {
	switch x := m.rand(100); {
	case x < 90:
		return m.rand(min(8, n)) + 1
	case x < 99:
		return m.rand(min(32, n)) + 1
	default:
		return m.rand(n) + 1
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// mutate performs several mutations on the provided values.
func (m *mutator) mutate(vals []any, maxBytes int) {
	// TODO(katiehockman): pull some of these functions into helper methods and
	// test that each case is working as expected.
	// TODO(katiehockman): perform more types of mutations for []byte.

	// maxPerVal will represent the maximum number of bytes that each value be
	// allowed after mutating, giving an equal amount of capacity to each line.
	// Allow a little wiggle room for the encoding.
	maxPerVal := maxBytes/len(vals) - 100

	// Pick a random value to mutate.
	// TODO: consider mutating more than one value at a time.
	i := m.rand(len(vals))
	switch v := vals[i].(type) {
	case int:
		vals[i] = int(m.mutateInt(int64(v), maxInt))
	case int8:
		vals[i] = int8(m.mutateInt(int64(v), math.MaxInt8))
	case int16:
		vals[i] = int16(m.mutateInt(int64(v), math.MaxInt16))
	case int64:
		vals[i] = m.mutateInt(v, maxInt)
	case uint:
		vals[i] = uint(m.mutateUInt(uint64(v), maxUint))
	case uint16:
		vals[i] = uint16(m.mutateUInt(uint64(v), math.MaxUint16))
	case uint32:
		vals[i] = uint32(m.mutateUInt(uint64(v), math.MaxUint32))
	case uint64:
		vals[i] = m.mutateUInt(uint64(v), maxUint)
	case float32:
		vals[i] = float32(m.mutateFloat(float64(v), math.MaxFloat32))
	case float64:
		vals[i] = m.mutateFloat(v, math.MaxFloat64)
	case bool:
		if m.rand(2) == 1 {
			vals[i] = !v // 50% chance of flipping the bool
		}
	case rune: // int32
		vals[i] = rune(m.mutateInt(int64(v), math.MaxInt32))
	case byte: // uint8
		vals[i] = byte(m.mutateUInt(uint64(v), math.MaxUint8))
	case string:
		if len(v) > maxPerVal {
			panic(fmt.Sprintf("cannot mutate bytes of length %d", len(v)))
		}
		if cap(m.scratch) < maxPerVal {
			m.scratch = append(make([]byte, 0, maxPerVal), v...)
		} else {
			m.scratch = m.scratch[:len(v)]
			copy(m.scratch, v)
		}
		m.mutateBytes(&m.scratch)
		vals[i] = string(m.scratch)
	case []byte:
		if len(v) > maxPerVal {
			panic(fmt.Sprintf("cannot mutate bytes of length %d", len(v)))
		}
		if cap(m.scratch) < maxPerVal {
			m.scratch = append(make([]byte, 0, maxPerVal), v...)
		} else {
			m.scratch = m.scratch[:len(v)]
			copy(m.scratch, v)
		}
		m.mutateBytes(&m.scratch)
		vals[i] = m.scratch
	default:
		panic(fmt.Sprintf("type not supported for mutating: %T", vals[i]))
	}
}

func (m *mutator) mutateInt(v, maxValue int64) int64 {
	var max int64
	for {
		max = 100
		switch m.rand(2) {
		case 0:
			// Add a random number
			if v >= maxValue {
				continue
			}
			if v > 0 && maxValue-v < max {
				// Don't let v exceed maxValue
				max = maxValue - v
			}
			v += int64(1 + m.rand(int(max)))
			return v
		case 1:
			// Subtract a random number
			if v <= -maxValue {
				continue
			}
			if v < 0 && maxValue+v < max {
				// Don't let v drop below -maxValue
				max = maxValue + v
			}
			v -= int64(1 + m.rand(int(max)))
			return v
		}
	}
}

func (m *mutator) mutateUInt(v, maxValue uint64) uint64 {
	var max uint64
	for {
		max = 100
		switch m.rand(2) {
		case 0:
			// Add a random number
			if v >= maxValue {
				continue
			}
			if v > 0 && maxValue-v < max {
				// Don't let v exceed maxValue
				max = maxValue - v
			}

			v += uint64(1 + m.rand(int(max)))
			return v
		case 1:
			// Subtract a random number
			if v <= 0 {
				continue
			}
			if v < max {
				// Don't let v drop below 0
				max = v
			}
			v -= uint64(1 + m.rand(int(max)))
			return v
		}
	}
}

func (m *mutator) mutateFloat(v, maxValue float64) float64 {
	var max float64
	for {
		switch m.rand(4) {
		case 0:
			// Add a random number
			if v >= maxValue {
				continue
			}
			max = 100
			if v > 0 && maxValue-v < max {
				// Don't let v exceed maxValue
				max = maxValue - v
			}
			v += float64(1 + m.rand(int(max)))
			return v
		case 1:
			// Subtract a random number
			if v <= -maxValue {
				continue
			}
			max = 100
			if v < 0 && maxValue+v < max {
				// Don't let v drop below -maxValue
				max = maxValue + v
			}
			v -= float64(1 + m.rand(int(max)))
			return v
		case 2:
			// Multiply by a random number
			absV := math.Abs(v)
			if v == 0 || absV >= maxValue {
				continue
			}
			max = 10
			if maxValue/absV < max {
				// Don't let v go beyond the minimum or maximum value
				max = maxValue / absV
			}
			v *= float64(1 + m.rand(int(max)))
			return v
		case 3:
			// Divide by a random number
			if v == 0 {
				continue
			}
			v /= float64(1 + m.rand(10))
			return v
		}
	}
}

type byteSliceMutator func(*mutator, []byte) []byte

var byteSliceMutators = []byteSliceMutator{
	byteSliceRemoveBytes,
	byteSliceInsertRandomBytes,
	byteSliceDuplicateBytes,
	byteSliceOverwriteBytes,
	byteSliceBitFlip,
	byteSliceXORByte,
	byteSliceSwapByte,
	byteSliceArithmeticUint8,
	byteSliceArithmeticUint16,
	byteSliceArithmeticUint32,
	byteSliceArithmeticUint64,
	byteSliceOverwriteInterestingUint8,
	byteSliceOverwriteInterestingUint16,
	byteSliceOverwriteInterestingUint32,
	byteSliceInsertConstantBytes,
	byteSliceOverwriteConstantBytes,
	byteSliceShuffleBytes,
	byteSliceSwapBytes,
}

func (m *mutator) mutateBytes(ptrB *[]byte) {
	b := *ptrB
	defer func() {
		oldHdr := (*reflect.SliceHeader)(unsafe.Pointer(ptrB))
		newHdr := (*reflect.SliceHeader)(unsafe.Pointer(&b))
		if oldHdr.Data != newHdr.Data {
			panic("data moved to new address")
		}
		*ptrB = b
	}()

	for {
		mut := byteSliceMutators[m.rand(len(byteSliceMutators))]
		if mutated := mut(m, b); mutated != nil {
			b = mutated
			return
		}
	}
}

var (
	interesting8  = []int8{-128, -1, 0, 1, 16, 32, 64, 100, 127}
	interesting16 = []int16{-32768, -129, 128, 255, 256, 512, 1000, 1024, 4096, 32767}
	interesting32 = []int32{-2147483648, -100663046, -32769, 32768, 65535, 65536, 100663045, 2147483647}
)

const (
	maxUint = uint64(^uint(0))
	maxInt  = int64(maxUint >> 1)
)

func init() {
	for _, v := range interesting8 {
		interesting16 = append(interesting16, int16(v))
	}
	for _, v := range interesting16 {
		interesting32 = append(interesting32, int32(v))
	}
}
