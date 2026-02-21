// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing

import (
	"bytes"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
)

// Big-endian encoding functions to avoid importing encoding/binary
// which would create an import cycle (encoding/binary tests import testing).

func beUint16(b []byte) uint16 {
	return uint16(b[0])<<8 | uint16(b[1])
}

func beUint32(b []byte) uint32 {
	return uint32(b[0])<<24 | uint32(b[1])<<16 | uint32(b[2])<<8 | uint32(b[3])
}

func beUint64(b []byte) uint64 {
	return uint64(b[0])<<56 | uint64(b[1])<<48 | uint64(b[2])<<40 | uint64(b[3])<<32 |
		uint64(b[4])<<24 | uint64(b[5])<<16 | uint64(b[6])<<8 | uint64(b[7])
}

func bePutUint16(b []byte, v uint16) {
	b[0] = byte(v >> 8)
	b[1] = byte(v)
}

func bePutUint32(b []byte, v uint32) {
	b[0] = byte(v >> 24)
	b[1] = byte(v >> 16)
	b[2] = byte(v >> 8)
	b[3] = byte(v)
}

func bePutUint64(b []byte, v uint64) {
	b[0] = byte(v >> 56)
	b[1] = byte(v >> 48)
	b[2] = byte(v >> 40)
	b[3] = byte(v >> 32)
	b[4] = byte(v >> 24)
	b[5] = byte(v >> 16)
	b[6] = byte(v >> 8)
	b[7] = byte(v)
}

// byteReader reads typed values from a byte slice.
// Used for deserializing libFuzzer input format.
type byteReader struct {
	data []byte
	pos  int
}

func newByteReader(data []byte) *byteReader {
	return &byteReader{data: data}
}

// remaining returns the number of unread bytes.
func (r *byteReader) remaining() int {
	if r.pos >= len(r.data) {
		return 0
	}
	return len(r.data) - r.pos
}

// readBytes returns the next n bytes, advancing the position.
// If fewer than n bytes remain, returns what's available.
func (r *byteReader) readBytes(n int) []byte {
	if n <= 0 {
		return nil
	}
	if r.pos >= len(r.data) {
		return nil
	}
	end := r.pos + n
	if end > len(r.data) {
		end = len(r.data)
	}
	b := r.data[r.pos:end]
	r.pos = end
	return b
}

// readBytesPadded returns exactly n bytes, zero-padding if necessary.
func (r *byteReader) readBytesPadded(n int) []byte {
	b := r.readBytes(n)
	if len(b) == n {
		return b
	}
	// Zero-pad
	padded := make([]byte, n)
	copy(padded, b)
	return padded
}

// readUint32 reads a big-endian uint32.
func (r *byteReader) readUint32() uint32 {
	b := r.readBytesPadded(4)
	return beUint32(b)
}

// readUint64 reads a big-endian uint64.
func (r *byteReader) readUint64() uint64 {
	b := r.readBytesPadded(8)
	return beUint64(b)
}

// fixedTypeSize returns the serialized size of a type,
// or 0 for dynamic types (string, []byte).
func fixedTypeSize(t reflect.Type) int {
	switch t.Kind() {
	case reflect.Bool, reflect.Int8, reflect.Uint8:
		return 1
	case reflect.Int16, reflect.Uint16:
		return 2
	case reflect.Int32, reflect.Uint32, reflect.Float32:
		return 4
	case reflect.Int, reflect.Int64, reflect.Uint, reflect.Uint64, reflect.Float64:
		return 8
	default:
		return 0
	}
}

// readFixedValue reads a fixed-size value of the given type from the reader.
func (r *byteReader) readFixedValue(t reflect.Type) any {
	switch t.Kind() {
	case reflect.Bool:
		b := r.readBytesPadded(1)
		return b[0] != 0
	case reflect.Int8:
		b := r.readBytesPadded(1)
		return int8(b[0])
	case reflect.Int16:
		return int16(beUint16(r.readBytesPadded(2)))
	case reflect.Int32:
		return int32(beUint32(r.readBytesPadded(4)))
	case reflect.Int:
		return int(beUint64(r.readBytesPadded(8)))
	case reflect.Int64:
		return int64(beUint64(r.readBytesPadded(8)))
	case reflect.Uint8:
		b := r.readBytesPadded(1)
		return b[0]
	case reflect.Uint16:
		return beUint16(r.readBytesPadded(2))
	case reflect.Uint32:
		return beUint32(r.readBytesPadded(4))
	case reflect.Uint:
		return uint(beUint64(r.readBytesPadded(8)))
	case reflect.Uint64:
		return beUint64(r.readBytesPadded(8))
	case reflect.Float32:
		return math.Float32frombits(beUint32(r.readBytesPadded(4)))
	case reflect.Float64:
		return math.Float64frombits(beUint64(r.readBytesPadded(8)))
	default:
		return nil
	}
}

// deserializeLibfuzzerBytes converts
// raw libFuzzer bytes to typed Go values.
//
// Format: [fixed-size args][uint32 weights...][dynamic data concatenated]
//
// For single []byte or string argument,
// all bytes are used directly.
// For multiple dynamic arguments, uint32
// weights determine proportional allocation.
func deserializeLibfuzzerBytes(
	data []byte,
	types []reflect.Type,
) []any {
	if len(types) == 0 {
		return nil
	}

	// Special case: single dynamic argument gets all data
	if len(types) == 1 {
		switch types[0].Kind() {
		case reflect.Slice:
			if types[0].Elem().Kind() == reflect.Uint8 {
				return []any{append([]byte(nil), data...)}
			}
		case reflect.String:
			return []any{string(data)}
		}
	}

	r := newByteReader(data)
	vals := make([]any, len(types))

	// First pass: read fixed-size arguments,
	// identify dynamic arguments
	var dynamicIndices []int
	for i, t := range types {
		if fixedTypeSize(t) > 0 {
			vals[i] = r.readFixedValue(t)
		} else {
			dynamicIndices = append(dynamicIndices, i)
		}
	}

	numDynamic := len(dynamicIndices)
	if numDynamic == 0 {
		return vals
	}

	// Single dynamic argument gets all remaining bytes
	if numDynamic == 1 {
		idx := dynamicIndices[0]
		chunk := r.readBytes(r.remaining())
		vals[idx] = dynamicValueFromBytes(types[idx], chunk)
		return vals
	}

	// Multiple dynamic arguments: read
	// weights and allocate proportionally
	weights := make([]uint32, numDynamic)
	var totalWeight uint64
	for i := range weights {
		weights[i] = r.readUint32()
		if weights[i] == 0 {
			weights[i] = 1
		}
		totalWeight += uint64(weights[i])
	}

	// Allocate remaining bytes based on weights
	remaining := r.remaining()
	for i, idx := range dynamicIndices {
		var size int
		if i == numDynamic-1 {
			// Last dynamic arg gets all remaining
			size = r.remaining()
		} else if totalWeight > 0 {
			size = int(uint64(remaining) * uint64(weights[i]) / totalWeight)
			totalWeight -= uint64(weights[i])
			remaining -= size
		}
		chunk := r.readBytes(size)
		vals[idx] = dynamicValueFromBytes(types[idx], chunk)
	}

	return vals
}

// dynamicValueFromBytes creates a string or []byte from raw bytes.
func dynamicValueFromBytes(t reflect.Type, data []byte) any {
	switch t.Kind() {
	case reflect.String:
		return string(data)
	case reflect.Slice:
		if t.Elem().Kind() == reflect.Uint8 {
			return append([]byte(nil), data...)
		}
	}
	return nil
}

// serializeLibfuzzerBytes converts typed Go
// values to libFuzzer byte format.
// This is the inverse of deserializeLibfuzzerBytes.
// Format: [fixed-size args][uint32 weights...][dynamic data concatenated]
func serializeLibfuzzerBytes(vals []any) []byte {
	var fixedBuf bytes.Buffer
	var dynamicData [][]byte

	var tmp [8]byte
	for _, v := range vals {
		switch x := v.(type) {
		case bool:
			if x {
				fixedBuf.WriteByte(1)
			} else {
				fixedBuf.WriteByte(0)
			}
		case int8:
			fixedBuf.WriteByte(byte(x))
		case int16:
			bePutUint16(tmp[:2], uint16(x))
			fixedBuf.Write(tmp[:2])
		case int32:
			bePutUint32(tmp[:4], uint32(x))
			fixedBuf.Write(tmp[:4])
		case int64:
			bePutUint64(tmp[:8], uint64(x))
			fixedBuf.Write(tmp[:8])
		case int:
			bePutUint64(tmp[:8], uint64(x))
			fixedBuf.Write(tmp[:8])
		case uint8:
			fixedBuf.WriteByte(x)
		case uint16:
			bePutUint16(tmp[:2], x)
			fixedBuf.Write(tmp[:2])
		case uint32:
			bePutUint32(tmp[:4], x)
			fixedBuf.Write(tmp[:4])
		case uint64:
			bePutUint64(tmp[:8], x)
			fixedBuf.Write(tmp[:8])
		case uint:
			bePutUint64(tmp[:8], uint64(x))
			fixedBuf.Write(tmp[:8])
		case float32:
			bePutUint32(tmp[:4], math.Float32bits(x))
			fixedBuf.Write(tmp[:4])
		case float64:
			bePutUint64(tmp[:8], math.Float64bits(x))
			fixedBuf.Write(tmp[:8])
		case string:
			dynamicData = append(dynamicData, []byte(x))
		case []byte:
			dynamicData = append(dynamicData, x)
		default:
			panic(fmt.Sprintf("serializeLibfuzzerBytes: unsupported type %T", v))
		}
	}

	// Write uint32 weights for multiple
	// dynamic args.
	// Use length as weight, but use 1
	// for empty data (matches deserialization
	// which treats zero weights as 1)
	if len(dynamicData) > 1 {
		var wtmp [4]byte
		for _, d := range dynamicData {
			weight := uint32(len(d))
			if weight == 0 {
				weight = 1
			}
			bePutUint32(wtmp[:], weight)
			fixedBuf.Write(wtmp[:])
		}
	}

	// Write dynamic data
	for _, d := range dynamicData {
		fixedBuf.Write(d)
	}

	return fixedBuf.Bytes()
}

// writeCorpusFile writes corpus data to a file with a hash-based name.
func writeCorpusFile(dir string, data []byte) error {
	name := fmt.Sprintf("%016x", fnv64a(data))
	path := filepath.Join(dir, name)

	// Skip if file already exists (duplicate content)
	if _, err := os.Stat(path); err == nil {
		return nil
	}

	return os.WriteFile(path, data, 0666)
}

// libFuzzer Corpus Reading
// readLibfuzzerCorpus reads all files from a libFuzzer corpus directory,
// deserializes them according to the provided types, and returns corpus entries.
// This enables using libFuzzer-generated corpus files with Go's native fuzzing.
func readLibfuzzerCorpus(dir string, types []reflect.Type) ([]corpusEntry, error) {
	files, err := os.ReadDir(dir)
	if os.IsNotExist(err) {
		return nil, nil // No corpus to read
	} else if err != nil {
		return nil, fmt.Errorf("reading libfuzzer corpus directory: %v", err)
	}

	var corpus []corpusEntry
	var errs []string
	for _, file := range files {
		if file.IsDir() {
			continue
		}
		filename := filepath.Join(dir, file.Name())
		data, err := os.ReadFile(filename)
		if err != nil {
			errs = append(errs, fmt.Sprintf("%s: read error: %v", file.Name(), err))
			continue
		}

		// Deserialize libFuzzer bytes to typed values
		vals := deserializeLibfuzzerBytes(data, types)
		corpus = append(corpus, corpusEntry{Path: filename, Values: vals})
	}

	if len(errs) > 0 {
		return corpus, fmt.Errorf("errors reading libfuzzer corpus:\n%s",
			strings.Join(errs, "\n"))
	}
	return corpus, nil
}

// convertLibfuzzerCorpus reads all files from a libFuzzer corpus directory,
// deserializes them according to the provided types, and writes them as
// Go-format corpus files to testdata/fuzz/<fuzzName>/.
func convertLibfuzzerCorpus(srcDir, fuzzName string, types []reflect.Type) error {
	files, err := os.ReadDir(srcDir)
	if os.IsNotExist(err) {
		return fmt.Errorf("libfuzzer corpus directory does not exist: %s", srcDir)
	} else if err != nil {
		return fmt.Errorf("reading libfuzzer corpus directory: %v", err)
	}

	// Create destination directory
	destDir := filepath.Join(corpusDir, fuzzName)
	if err := os.MkdirAll(destDir, 0777); err != nil {
		return fmt.Errorf("creating corpus directory: %v", err)
	}

	var converted, skipped int
	var errs []string
	for _, file := range files {
		if file.IsDir() {
			continue
		}
		filename := filepath.Join(srcDir, file.Name())
		data, err := os.ReadFile(filename)
		if err != nil {
			errs = append(errs, fmt.Sprintf("%s: read error: %v", file.Name(), err))
			continue
		}

		// Deserialize libFuzzer bytes to typed values
		vals := deserializeLibfuzzerBytes(data, types)
		if vals == nil {
			skipped++
			continue
		}

		// Marshal to Go corpus format
		corpusData := marshalGoCorpusFile(vals)

		// Write with hash-based filename
		if err := writeCorpusFile(destDir, corpusData); err != nil {
			errs = append(errs, fmt.Sprintf("%s: write error: %v", file.Name(), err))
			continue
		}
		converted++
	}

	fmt.Printf("Converted %d corpus files to %s\n", converted, destDir)
	if skipped > 0 {
		fmt.Printf("Skipped %d files that could not be deserialized\n", skipped)
	}

	if len(errs) > 0 {
		return fmt.Errorf("errors converting libfuzzer corpus:\n%s", strings.Join(errs, "\n"))
	}
	return nil
}

// libFuzzer Iteration Runner
// runFuzzIteration executes a fuzz function with the given input data.
// This is the core logic for running a single fuzz iteration, separated from
// the libfuzzer-specific setup code so it can be unit tested.
func runFuzzIteration(
	name string,
	fn func(*F),
	fstate *fuzzState,
	data []byte,
) {
	// Set the input data for this invocation
	fstate.libfuzzerInput = data

	// Create a minimal F for the fuzz function
	f := &F{
		common: common{
			name:   name,
			signal: make(chan bool),
			w:      os.Stderr,
		},
		fstate: fstate,
	}

	// Run the fuzz function in a goroutine so that f.Skip()
	// can use runtime.Goexit() properly (can't call Goexit from C thread)
	done := make(chan bool)
	var panicked interface{}
	go func() {
		defer func() {
			panicked = recover()
			close(done)
		}()
		fn(f)
	}()
	<-done

	// Run any cleanup functions registered with f.Cleanup()
	f.runCleanup(normalPanic)

	// Handle panics from the fuzz function - re-panic to signal crash to libFuzzer
	if panicked != nil {
		panic(panicked)
	}

	// Check if the fuzz test was skipped via f.Skip() or f.SkipNow()
	if f.skipped {
		return
	}

	// Check if the fuzz test failed via f.Fatal() or f.FailNow() at F level.
	// These call runtime.Goexit() so they don't trigger a panic.
	if f.failed {
		panic("testing: fuzz test failed before Fuzz() was called")
	}
}
