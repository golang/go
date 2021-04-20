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
func (m *mutator) mutate(vals []interface{}, maxBytes int) {
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
		// TODO(jayconrod,katiehockman): Keep a []byte somewhere (maybe in
		// mutator) that we mutate repeatedly to avoid re-allocating the data
		// every time.
		if len(v) > maxPerVal {
			panic(fmt.Sprintf("cannot mutate bytes of length %d", len(v)))
		}
		b := []byte(v)
		if cap(b) < maxPerVal {
			b = append(make([]byte, 0, maxPerVal), b...)
		}
		m.mutateBytes(&b)
		vals[i] = string(b)
	case []byte:
		if len(v) > maxPerVal {
			panic(fmt.Sprintf("cannot mutate bytes of length %d", len(v)))
		}
		if cap(v) < maxPerVal {
			v = append(make([]byte, 0, maxPerVal), v...)
		}
		m.mutateBytes(&v)
		vals[i] = v
	default:
		panic(fmt.Sprintf("type not supported for mutating: %T", vals[i]))
	}
}

func (m *mutator) mutateInt(v, maxValue int64) int64 {
	numIters := 1 + m.r.exp2()
	var max int64
	for iter := 0; iter < numIters; iter++ {
		max = 100
		switch m.rand(2) {
		case 0:
			// Add a random number
			if v >= maxValue {
				iter--
				continue
			}
			if v > 0 && maxValue-v < max {
				// Don't let v exceed maxValue
				max = maxValue - v
			}
			v += int64(1 + m.rand(int(max)))
		case 1:
			// Subtract a random number
			if v <= -maxValue {
				iter--
				continue
			}
			if v < 0 && maxValue+v < max {
				// Don't let v drop below -maxValue
				max = maxValue + v
			}
			v -= int64(1 + m.rand(int(max)))
		}
	}
	return v
}

func (m *mutator) mutateUInt(v, maxValue uint64) uint64 {
	numIters := 1 + m.r.exp2()
	var max uint64
	for iter := 0; iter < numIters; iter++ {
		max = 100
		switch m.rand(2) {
		case 0:
			// Add a random number
			if v >= maxValue {
				iter--
				continue
			}
			if v > 0 && maxValue-v < max {
				// Don't let v exceed maxValue
				max = maxValue - v
			}

			v += uint64(1 + m.rand(int(max)))
		case 1:
			// Subtract a random number
			if v <= 0 {
				iter--
				continue
			}
			if v < max {
				// Don't let v drop below 0
				max = v
			}
			v -= uint64(1 + m.rand(int(max)))
		}
	}
	return v
}

func (m *mutator) mutateFloat(v, maxValue float64) float64 {
	numIters := 1 + m.r.exp2()
	var max float64
	for iter := 0; iter < numIters; iter++ {
		switch m.rand(4) {
		case 0:
			// Add a random number
			if v >= maxValue {
				iter--
				continue
			}
			max = 100
			if v > 0 && maxValue-v < max {
				// Don't let v exceed maxValue
				max = maxValue - v
			}
			v += float64(1 + m.rand(int(max)))
		case 1:
			// Subtract a random number
			if v <= -maxValue {
				iter--
				continue
			}
			max = 100
			if v < 0 && maxValue+v < max {
				// Don't let v drop below -maxValue
				max = maxValue + v
			}
			v -= float64(1 + m.rand(int(max)))
		case 2:
			// Multiply by a random number
			absV := math.Abs(v)
			if v == 0 || absV >= maxValue {
				iter--
				continue
			}
			max = 10
			if maxValue/absV < max {
				// Don't let v go beyond the minimum or maximum value
				max = maxValue / absV
			}
			v *= float64(1 + m.rand(int(max)))
		case 3:
			// Divide by a random number
			if v == 0 {
				iter--
				continue
			}
			v /= float64(1 + m.rand(10))
		}
	}
	return v
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
		switch m.rand(18) {
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
			n := m.chooseLen(1024)
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
			// Duplicate a range of bytes and insert it into
			// a random position
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
			// Overwrite a range of bytes with a randomly selected
			// chunk
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
			// In order to avoid a no-op (where the random value matches
			// the existing value), use XOR instead of just setting to
			// the random value.
			b[pos] ^= byte(1 + m.rand(255))
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
		case 14:
			// Insert a range of constant bytes.
			if len(b) <= 1 {
				iter--
				continue
			}
			dst := m.rand(len(b))
			// TODO(rolandshoemaker,katiehockman): 4096 was mainly picked
			// randomly. We may want to either pick a much larger value
			// (AFL uses 32768, paired with a similar impl to chooseLen
			// which biases towards smaller lengths that grow over time),
			// or set the max based on characteristics of the corpus
			// (libFuzzer sets a min/max based on the min/max size of
			// entries in the corpus and then picks uniformly from
			// that range).
			n := m.chooseLen(4096)
			if len(b)+n >= cap(b) {
				iter--
				continue
			}
			b = b[:len(b)+n]
			copy(b[dst+n:], b[dst:])
			rb := byte(m.rand(256))
			for i := dst; i < dst+n; i++ {
				b[i] = rb
			}
		case 15:
			// Overwrite a range of bytes with a chunk of
			// constant bytes.
			if len(b) <= 1 {
				iter--
				continue
			}
			dst := m.rand(len(b))
			n := m.chooseLen(len(b) - dst)
			rb := byte(m.rand(256))
			for i := dst; i < dst+n; i++ {
				b[i] = rb
			}
		case 16:
			// Shuffle a range of bytes
			if len(b) <= 1 {
				iter--
				continue
			}
			dst := m.rand(len(b))
			n := m.chooseLen(len(b) - dst)
			if n <= 2 {
				iter--
				continue
			}
			// Start at the end of the range, and iterate backwards
			// to dst, swapping each element with another element in
			// dst:dst+n (Fisher-Yates shuffle).
			for i := n - 1; i > 0; i-- {
				j := m.rand(i + 1)
				b[dst+i], b[dst+j] = b[dst+j], b[dst+i]
			}
		case 17:
			// Swap two chunks
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
			tmp := make([]byte, n)
			copy(tmp, b[dst:])
			copy(b[dst:], b[src:src+n])
			copy(b[src:], tmp)
		default:
			panic("unknown mutator")
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
