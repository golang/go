// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fuzz

import (
	"encoding/binary"
	"fmt"
	"reflect"
	"unsafe"
)

type mutator struct {
	r *pcgRand
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

// chooseLen chooses length of range mutation in range [0,n]. It gives
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
func (m *mutator) mutate(vals []interface{}, maxBytes int) []interface{} {
	// TODO(jayconrod,katiehockman): use as few allocations as possible
	// TODO(katiehockman): pull some of these functions into helper methods and
	// test that each case is working as expected.
	// TODO(katiehockman): perform more types of mutations.

	// maxPerVal will represent the maximum number of bytes that each value be
	// allowed after mutating, giving an equal amount of capacity to each line.
	// Allow a little wiggle room for the encoding.
	maxPerVal := maxBytes/len(vals) - 100

	// Pick a random value to mutate.
	// TODO: consider mutating more than one value at a time.
	i := m.rand(len(vals))
	// TODO(katiehockman): support mutating other types
	switch v := vals[i].(type) {
	case []byte:
		if len(v) > maxPerVal {
			panic(fmt.Sprintf("cannot mutate bytes of length %d", len(v)))
		}
		b := make([]byte, 0, maxPerVal)
		b = append(b, v...)
		m.mutateBytes(&b)
		vals[i] = b
		return vals
	default:
		panic(fmt.Sprintf("type not supported for mutating: %T", vals[i]))
	}
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

	numIters := 1 + m.r.exp2()
	for iter := 0; iter < numIters; iter++ {
		switch m.rand(10) {
		case 0:
			// Remove a range of bytes.
			if len(b) <= 1 {
				iter--
				continue
			}
			pos0 := m.rand(len(b))
			pos1 := pos0 + m.chooseLen(len(b)-pos0)
			copy(b[pos0:], b[pos1:])
			b = b[:len(b)-(pos1-pos0)]
		case 1:
			// Insert a range of random bytes.
			pos := m.rand(len(b) + 1)
			n := m.chooseLen(10)
			if len(b)+n >= cap(b) {
				iter--
				continue
			}
			b = b[:len(b)+n]
			copy(b[pos+n:], b[pos:])
			for i := 0; i < n; i++ {
				b[pos+i] = byte(m.rand(256))
			}
		case 2:
			// Duplicate a range of bytes.
			if len(b) <= 1 {
				iter--
				continue
			}
			src := m.rand(len(b))
			dst := m.rand(len(b))
			for dst == src {
				dst = m.rand(len(b))
			}
			n := m.chooseLen(len(b) - src)
			if len(b)+n >= cap(b) {
				iter--
				continue
			}
			tmp := make([]byte, n)
			copy(tmp, b[src:])
			b = b[:len(b)+n]
			copy(b[dst+n:], b[dst:])
			copy(b[dst:], tmp)
		case 3:
			// Copy a range of bytes.
			if len(b) <= 1 {
				iter--
				continue
			}
			src := m.rand(len(b))
			dst := m.rand(len(b))
			for dst == src {
				dst = m.rand(len(b))
			}
			n := m.chooseLen(len(b) - src)
			copy(b[dst:], b[src:src+n])
		case 4:
			// Bit flip.
			if len(b) == 0 {
				iter--
				continue
			}
			pos := m.rand(len(b))
			b[pos] ^= 1 << uint(m.rand(8))
		case 5:
			// Set a byte to a random value.
			if len(b) == 0 {
				iter--
				continue
			}
			pos := m.rand(len(b))
			b[pos] = byte(m.rand(256))
		case 6:
			// Swap 2 bytes.
			if len(b) <= 1 {
				iter--
				continue
			}
			src := m.rand(len(b))
			dst := m.rand(len(b))
			for dst == src {
				dst = m.rand(len(b))
			}
			b[src], b[dst] = b[dst], b[src]
		case 7:
			// Add/subtract from a byte.
			if len(b) == 0 {
				iter--
				continue
			}
			pos := m.rand(len(b))
			v := byte(m.rand(35) + 1)
			if m.r.bool() {
				b[pos] += v
			} else {
				b[pos] -= v
			}
		case 8:
			// Add/subtract from a uint16.
			if len(b) < 2 {
				iter--
				continue
			}
			v := uint16(m.rand(35) + 1)
			if m.r.bool() {
				v = 0 - v
			}
			pos := m.rand(len(b) - 1)
			enc := m.randByteOrder()
			enc.PutUint16(b[pos:], enc.Uint16(b[pos:])+v)
		case 9:
			// Add/subtract from a uint32.
			if len(b) < 4 {
				iter--
				continue
			}
			v := uint32(m.rand(35) + 1)
			if m.r.bool() {
				v = 0 - v
			}
			pos := m.rand(len(b) - 3)
			enc := m.randByteOrder()
			enc.PutUint32(b[pos:], enc.Uint32(b[pos:])+v)
		case 10:
			// Add/subtract from a uint64.
			if len(b) < 8 {
				iter--
				continue
			}
			v := uint64(m.rand(35) + 1)
			if m.r.bool() {
				v = 0 - v
			}
			pos := m.rand(len(b) - 7)
			enc := m.randByteOrder()
			enc.PutUint64(b[pos:], enc.Uint64(b[pos:])+v)
		case 11:
			// Replace a byte with an interesting value.
			if len(b) == 0 {
				iter--
				continue
			}
			pos := m.rand(len(b))
			b[pos] = byte(interesting8[m.rand(len(interesting8))])
		case 12:
			// Replace a uint16 with an interesting value.
			if len(b) < 2 {
				iter--
				continue
			}
			pos := m.rand(len(b) - 1)
			v := uint16(interesting16[m.rand(len(interesting16))])
			m.randByteOrder().PutUint16(b[pos:], v)
		case 13:
			// Replace a uint32 with an interesting value.
			if len(b) < 4 {
				iter--
				continue
			}
			pos := m.rand(len(b) - 3)
			v := uint32(interesting32[m.rand(len(interesting32))])
			m.randByteOrder().PutUint32(b[pos:], v)
		}
	}
}

var (
	interesting8  = []int8{-128, -1, 0, 1, 16, 32, 64, 100, 127}
	interesting16 = []int16{-32768, -129, 128, 255, 256, 512, 1000, 1024, 4096, 32767}
	interesting32 = []int32{-2147483648, -100663046, -32769, 32768, 65535, 65536, 100663045, 2147483647}
)

func init() {
	for _, v := range interesting8 {
		interesting16 = append(interesting16, int16(v))
	}
	for _, v := range interesting16 {
		interesting32 = append(interesting32, int32(v))
	}
}
