// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fuzz

import (
	"encoding/binary"
	"math"
	"reflect"
)

// Source consumes a byte slice and extracts typed arguments from it.
// This is used by libFuzzer mode to deserialize the raw byte input from libFuzzer
// into the typed arguments expected by Go fuzz functions.
type Source struct {
	data      []byte
	pos       int
	exhausted bool
}

// NewSource creates a new Source from the given byte slice.
func NewSource(data []byte) *Source {
	return &Source{data: data, pos: 0, exhausted: false}
}

// IsExhausted returns true if we tried to read more data than available.
func (s *Source) IsExhausted() bool {
	return s.exhausted
}

// Remaining returns the number of unread bytes.
func (s *Source) Remaining() int {
	if s.pos >= len(s.data) {
		return 0
	}
	return len(s.data) - s.pos
}

// consumeBytes reads exactly n bytes from the source.
// If fewer than n bytes remain, it returns a zero-padded slice
// and marks the source as exhausted.
func (s *Source) consumeBytes(n int) []byte {
	if s.pos+n <= len(s.data) {
		// Fast path: enough data available, return direct slice
		result := s.data[s.pos : s.pos+n]
		s.pos += n
		return result
	}

	// Slow path: not enough data, zero-pad the result
	s.exhausted = true
	result := make([]byte, n)
	remaining := len(s.data) - s.pos
	if remaining > 0 {
		copy(result, s.data[s.pos:])
		s.pos = len(s.data)
	}
	return result
}

// isFixedSizeKind returns true for types with fixed byte sizes (bool, integers, floats).
func isFixedSizeKind(k reflect.Kind) bool {
	return k >= reflect.Bool && k <= reflect.Float64
}

// readInt reads a signed integer of the given kind from the source.
func (s *Source) readInt(k reflect.Kind) int64 {
	switch k {
	case reflect.Int8:
		return int64(int8(s.consumeBytes(1)[0]))
	case reflect.Int16:
		return int64(int16(binary.BigEndian.Uint16(s.consumeBytes(2))))
	case reflect.Int32:
		return int64(int32(binary.BigEndian.Uint32(s.consumeBytes(4))))
	case reflect.Int64, reflect.Int:
		return int64(binary.BigEndian.Uint64(s.consumeBytes(8)))
	}
	return 0
}

// readUint reads an unsigned integer of the given kind from the source.
func (s *Source) readUint(k reflect.Kind) uint64 {
	switch k {
	case reflect.Uint8:
		return uint64(s.consumeBytes(1)[0])
	case reflect.Uint16:
		return uint64(binary.BigEndian.Uint16(s.consumeBytes(2)))
	case reflect.Uint32:
		return uint64(binary.BigEndian.Uint32(s.consumeBytes(4)))
	case reflect.Uint64, reflect.Uint:
		return binary.BigEndian.Uint64(s.consumeBytes(8))
	}
	return 0
}

// FillArgs creates fuzz function arguments from the byte source.
// The types parameter should be the types of the fuzz function parameters
// (excluding *testing.T).
//
// The algorithm:
//  1. Fixed-size types (bool, integers, floats) are filled first,
//     consuming their exact byte sizes.
//  2. Dynamic types (string, []byte) share the remaining bytes proportionally
//     based on weight bytes read from the input.
//     This allows libFuzzer to control the distribution.
func (s *Source) FillArgs(types []reflect.Type) []reflect.Value {
	args := make([]reflect.Value, len(types))
	dynamicIndices := s.fillFixedSizeArgs(args, types)

	if len(dynamicIndices) > 0 {
		s.fillDynamicArgs(args, types, dynamicIndices)
	}
	return args
}

// fillFixedSizeArgs fills all fixed-size arguments and returns indices of
// dynamic arguments.
func (s *Source) fillFixedSizeArgs(args []reflect.Value, types []reflect.Type) []int {
	var dynamicIndices []int
	for i, t := range types {
		if isFixedSizeKind(t.Kind()) {
			args[i] = s.readFixedValue(t)
		} else {
			dynamicIndices = append(dynamicIndices, i)
		}
	}
	return dynamicIndices
}

// fillDynamicArgs distributes remaining bytes among dynamic arguments
// (string, []byte).
func (s *Source) fillDynamicArgs(
	args []reflect.Value, types []reflect.Type, indices []int,
) {
	numDynamic := len(indices)

	// Single dynamic argument gets all remaining bytes
	if numDynamic == 1 {
		args[indices[0]] = s.readDynamicValue(types[indices[0]], s.Remaining())
		return
	}

	// Multiple dynamic arguments: use weight bytes for proportional allocation
	sizes := s.calculateWeightedSizes(numDynamic)
	for i, argIdx := range indices {
		args[argIdx] = s.readDynamicValue(types[argIdx], sizes[i])
	}
}

// calculateWeightedSizes reads weight bytes and calculates proportional sizes
// for dynamic arguments.
// The last argument receives all remaining bytes to avoid rounding errors.
func (s *Source) calculateWeightedSizes(count int) []int {
	weights := s.consumeBytes(count)
	totalWeight := sumBytes(weights)
	totalBytes := s.Remaining()

	sizes := make([]int, count)
	for i := 0; i < count-1; i++ {
		if totalWeight > 0 {
			sizes[i] = (totalBytes * int(weights[i])) / totalWeight
		} else {
			sizes[i] = totalBytes / count
		}
	}
	// Last argument gets whatever remains (assigned during read via Remaining())
	sizes[count-1] = -1 // sentinel: use Remaining() when reading
	return sizes
}

// sumBytes returns the sum of all bytes as an int.
func sumBytes(b []byte) int {
	sum := 0
	for _, v := range b {
		sum += int(v)
	}
	return sum
}

// readFixedValue reads a fixed-size value (bool, integer, or float).
func (s *Source) readFixedValue(t reflect.Type) reflect.Value {
	v := reflect.New(t).Elem()
	k := t.Kind()

	switch {
	case k >= reflect.Int && k <= reflect.Int64:
		v.SetInt(s.readInt(k))
	case k >= reflect.Uint && k <= reflect.Uint64:
		v.SetUint(s.readUint(k))
	case k == reflect.Float32:
		bits := uint32(s.readUint(reflect.Uint32))
		v.Set(reflect.ValueOf(math.Float32frombits(bits)))
	case k == reflect.Float64:
		bits := s.readUint(reflect.Uint64)
		v.Set(reflect.ValueOf(math.Float64frombits(bits)))
	case k == reflect.Bool:
		v.SetBool(s.consumeBytes(1)[0]&1 != 0)
	}
	return v
}

// readDynamicValue reads a dynamic-size value (string or []byte).
// If size is -1, it reads all remaining bytes.
func (s *Source) readDynamicValue(t reflect.Type, size int) reflect.Value {
	if size < 0 {
		size = s.Remaining()
	}

	v := reflect.New(t).Elem()
	switch t.Kind() {
	case reflect.String:
		v.SetString(string(s.consumeBytes(size)))
	case reflect.Slice:
		if t.Elem().Kind() == reflect.Uint8 {
			v.SetBytes(s.consumeBytes(size))
		}
	}
	return v
}
