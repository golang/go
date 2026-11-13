// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package upki

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"maps"
	"math"
	"os"
	"slices"
	"sync"
	"testing"

	"golang.org/x/crypto/cryptobyte"
)

func TestIndexLookup(t *testing.T) {
	t.Parallel()

	const (
		firstIntervalLow   timestamp = 500
		firstIntervalHigh  timestamp = 1500
		secondIntervalLow  timestamp = 2000
		secondIntervalHigh timestamp = 3000
		queryTime          timestamp = 1000
	)
	logA, logB := logID{0xaa}, logID{0xbb}
	idx := openTestIndex(t, encodeTestIndex([]testIndexFilter{
		{filename: "a.filter", coverage: []testIndexCoverage{{logID: logA, low: 100, high: 200}}},
		{filename: "b.filter", coverage: []testIndexCoverage{
			{logID: logB, low: firstIntervalLow, high: firstIntervalHigh},
			{logID: logB, low: secondIntervalLow, high: secondIntervalHigh},
		}},
	}))
	// Cover both binary-search misses and the inclusive timestamp bounds of
	// each interval recorded for a matching log.
	tests := []struct {
		name string
		log  logID
		time timestamp
		want []string
	}{
		{name: "missing log", log: logID{0xcc}, time: queryTime},
		{name: "below interval", log: logB, time: firstIntervalLow - 1},
		{name: "lower bound", log: logB, time: firstIntervalLow, want: []string{"b.filter"}},
		{name: "inside interval", log: logB, time: queryTime, want: []string{"b.filter"}},
		{name: "upper bound", log: logB, time: firstIntervalHigh, want: []string{"b.filter"}},
		{name: "between intervals", log: logB, time: firstIntervalHigh + 1},
		{name: "above interval", log: logB, time: secondIntervalHigh + 1},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			got, err := idx.lookup(test.log, test.time)
			if err != nil {
				t.Fatalf("lookup: %v", err)
			}
			if !slices.Equal(got, test.want) {
				t.Errorf("lookup = %q, want %q", got, test.want)
			}
		})
	}
}

func TestIndexLookupMultipleFilters(t *testing.T) {
	t.Parallel()

	const queryTime timestamp = 1000
	log := logID{1}
	// The middle entry misses queryTime. Lookup must skip it, continue through
	// the entry section, and preserve the order of both covering filters.
	idx := openTestIndex(t, encodeTestIndex([]testIndexFilter{
		{filename: "first.filter", coverage: []testIndexCoverage{{logID: log, low: 0, high: 2000}}},
		{filename: "miss.filter", coverage: []testIndexCoverage{{logID: log, low: 3000, high: 4000}}},
		{filename: "second.filter", coverage: []testIndexCoverage{{logID: log, low: 500, high: 1500}}},
	}))
	got, err := idx.lookup(log, queryTime)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"first.filter", "second.filter"}
	if !slices.Equal(got, want) {
		t.Fatalf("lookup = %q, want %q", got, want)
	}
}

func TestIndexMalformedEntry(t *testing.T) {
	t.Parallel()

	const queryTime timestamp = 150
	log := logID{0xaa}
	// An out-of-range filter index invalidates the whole entry section, even
	// when the corrupt entry's timestamp interval does not match the query.
	oneEntry := encodeTestIndex([]testIndexFilter{{
		filename: "a.filter",
		coverage: []testIndexCoverage{{logID: log, low: 100, high: 200}},
	}})
	oneEntry[len(oneEntry)-entrySize+1] = 1

	twoEntries := encodeTestIndex([]testIndexFilter{{
		filename: "a.filter",
		coverage: []testIndexCoverage{
			{logID: log, low: 100, high: 200},
			{logID: log, low: 300, high: 400},
		},
	}})
	twoEntries[len(twoEntries)-entrySize+1] = 1

	for _, test := range []struct {
		name string
		data []byte
	}{
		{"matching entry", oneEntry},
		{"nonmatching entry", twoEntries},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			idx := openTestIndex(t, test.data)
			if _, err := idx.lookup(log, queryTime); !errors.Is(err, errInvalidIndex) {
				t.Fatalf("lookup error = %v, want errInvalidIndex", err)
			}
		})
	}
}

func TestIndexLookupConcurrent(t *testing.T) {
	t.Parallel()

	// index documents that lookup is safe for concurrent use. Its tables are
	// immutable after construction and the shared *os.File is accessed only
	// with ReadAt. Exercise that contract across distinct entry sections. The
	// race-enabled test run checks the same access pattern for data races.
	const (
		logCount                   = 8
		workerCount                = 16
		lookupsPerWorker           = 50
		coverageLow      timestamp = 0
		coverageHigh     timestamp = 10
		queryTime        timestamp = 5
	)
	var filters []testIndexFilter
	for i := range logCount {
		filters = append(filters, testIndexFilter{
			filename: fmt.Sprintf("filter-%d", i),
			coverage: []testIndexCoverage{{logID: logID{byte(i + 1)}, low: coverageLow, high: coverageHigh}},
		})
	}
	idx := openTestIndex(t, encodeTestIndex(filters))

	var wg sync.WaitGroup
	errs := make(chan error, workerCount)
	for worker := range workerCount {
		wg.Go(func() {
			for iteration := range lookupsPerWorker {
				i := (worker + iteration) % len(filters)
				got, err := idx.lookup(logID{byte(i + 1)}, queryTime)
				if err != nil {
					errs <- err
					return
				}
				if want := []string{fmt.Sprintf("filter-%d", i)}; !slices.Equal(got, want) {
					errs <- fmt.Errorf("lookup = %q, want %q", got, want)
					return
				}
			}
		})
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}

func TestIndexFilenameValidation(t *testing.T) {
	t.Parallel()

	good := []string{
		"a",
		"...",
		"..hidden",
		"AZaz09-._",
		"1234567890123456789012345.filter",
	}
	bad := []string{
		"",
		".",
		"..",
		"../evil.filter",
		`..\evil.filter`,
		"a/b.filter",
		"/etc/passwd",
		"sp ace.filter",
		"tab\t.filter",
		"caf\xc3\xa9.filter",
		"high\xff.filter",
	}
	for _, name := range good {
		t.Run("accept "+name, func(t *testing.T) {
			t.Parallel()
			openTestIndex(t, encodeTestIndex([]testIndexFilter{{filename: name}}))
		})
	}
	for _, name := range bad {
		t.Run(fmt.Sprintf("reject %q", name), func(t *testing.T) {
			t.Parallel()
			_, err := newTestIndex(t, encodeTestIndex([]testIndexFilter{{filename: name}}))
			if !errors.Is(err, errInvalidIndex) {
				t.Fatalf("newIndex error = %v, want errInvalidIndex", err)
			}
		})
	}
}

func TestIndexMalformed(t *testing.T) {
	t.Parallel()

	logA, logB := logID{0xaa}, logID{0xbb}
	valid := encodeTestIndex([]testIndexFilter{{
		filename: "a.filter",
		coverage: []testIndexCoverage{{logID: logA, low: 100, high: 200}},
	}})
	badMagic := bytes.Clone(valid)
	copy(badMagic, "badmagic")
	tooManyLogs := make([]byte, headerSize)
	copy(tooManyLogs, indexMagic)
	binary.BigEndian.PutUint32(tooManyLogs[10:14], ^uint32(0))
	overlapsTables := bytes.Clone(valid)
	setTestIndexEntryOffset(overlapsTables, 1, 0, 0)
	sectionEndOverflow := bytes.Clone(valid)
	setTestIndexEntryOffset(sectionEndOverflow, 1, 0, math.MaxInt64)

	twoLogs := encodeTestIndex([]testIndexFilter{
		{filename: "a.filter", coverage: []testIndexCoverage{{logID: logA, low: 100, high: 200}}},
		{filename: "b.filter", coverage: []testIndexCoverage{{logID: logB, low: 100, high: 200}}},
	})
	unsorted := bytes.Clone(twoLogs)
	swapTestIndexLogEntries(unsorted, 2, 0, 1)
	duplicateLog := bytes.Clone(twoLogs)
	copy(testIndexLogID(duplicateLog, 2, 1), testIndexLogID(duplicateLog, 2, 0))
	overlappingSections := bytes.Clone(twoLogs)
	firstSection := testIndexEntryOffset(overlappingSections, 2, 0)
	setTestIndexEntryOffset(overlappingSections, 2, 1, firstSection+1)

	// Start from valid encodings and mutate one structural invariant per case,
	// so each rejection identifies the guard it is intended to exercise.
	tests := []struct {
		name string
		data []byte
	}{
		{"empty", nil},
		{"bad magic", badMagic},
		{"log count exceeds file", tooManyLogs},
		{"truncated header", valid[:headerSize-1]},
		{"truncated tables", valid[:headerSize+filenameSize-1]},
		{"entry section beyond EOF", valid[:len(valid)-1]},
		{"entry section end overflows int64", sectionEndOverflow},
		{"entry section overlaps tables", overlapsTables},
		{"unsorted log directory", unsorted},
		{"duplicate log", duplicateLog},
		{"overlapping entry sections", overlappingSections},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			_, err := newTestIndex(t, test.data)
			if !errors.Is(err, errInvalidIndex) {
				t.Fatalf("newIndex error = %v, want errInvalidIndex", err)
			}
		})
	}
}

func TestReadFullAt(t *testing.T) {
	t.Parallel()

	data := []byte("index data")
	t.Run("EOF with full read", func(t *testing.T) {
		t.Parallel()
		buf := make([]byte, len(data))
		n, err := readFullAt(eofReaderAt{bytes.NewReader(data)}, buf, 0)
		if err != nil || n != len(data) || !bytes.Equal(buf, data) {
			t.Fatalf("readFullAt = %d, %v, %q", n, err, buf)
		}
	})
	t.Run("short read without error", func(t *testing.T) {
		t.Parallel()
		buf := make([]byte, len(data))
		if _, err := readFullAt(shortReaderAt{bytes.NewReader(data)}, buf, 0); !errors.Is(err, io.ErrUnexpectedEOF) {
			t.Fatalf("readFullAt error = %v, want io.ErrUnexpectedEOF", err)
		}
	})
}

type testIndexFilter struct {
	filename string
	coverage []testIndexCoverage
}

type testIndexCoverage struct {
	logID     logID
	low, high timestamp
}

func encodeTestIndex(filters []testIndexFilter) []byte {
	// This mirrors the index.bin layout in C2SP upki-revocation v1.0.0:
	// <https://c2sp.org/upki-revocation@v1.0.0#the-index>.
	// It implements only the construction these reader tests need.
	type entry struct {
		filterIndex uint16
		low, high   timestamp
	}
	byLog := make(map[logID][]entry)
	for i, filter := range filters {
		if len(filter.filename) > filenameSize {
			panic("test index filename is too long")
		}
		for _, coverage := range filter.coverage {
			byLog[coverage.logID] = append(byLog[coverage.logID], entry{
				filterIndex: uint16(i), low: coverage.low, high: coverage.high,
			})
		}
	}
	logs := slices.SortedFunc(maps.Keys(byLog), func(a, b logID) int {
		return bytes.Compare(a[:], b[:])
	})

	var builder cryptobyte.Builder
	builder.AddBytes([]byte(indexMagic))
	builder.AddUint16(uint16(len(filters)))
	builder.AddUint32(uint32(len(logs)))
	for _, filter := range filters {
		slot := make([]byte, filenameSize)
		copy(slot, filter.filename)
		builder.AddBytes(slot)
	}
	offset := headerSize + filenameSize*len(filters) + logDirEntrySize*len(logs)
	for _, log := range logs {
		builder.AddBytes(log[:])
		builder.AddUint64(uint64(offset))
		builder.AddUint16(uint16(len(byLog[log])))
		offset += entrySize * len(byLog[log])
	}
	for _, log := range logs {
		for _, entry := range byLog[log] {
			builder.AddUint16(entry.filterIndex)
			builder.AddUint64(uint64(entry.low))
			builder.AddUint64(uint64(entry.high))
		}
	}

	return builder.BytesOrPanic()
}

func openTestIndex(t *testing.T, data []byte) *index {
	t.Helper()
	idx, err := newTestIndex(t, data)
	if err != nil {
		t.Fatal(err)
	}

	return idx
}

func newTestIndex(t *testing.T, data []byte) (*index, error) {
	t.Helper()
	file, err := os.CreateTemp(t.TempDir(), "index.bin")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { file.Close() })
	if _, err := file.Write(data); err != nil {
		t.Fatal(err)
	}

	return newIndex(file)
}

type eofReaderAt struct{ *bytes.Reader }

func (r eofReaderAt) ReadAt(buf []byte, offset int64) (int, error) {
	n, err := r.Reader.ReadAt(buf, offset)
	if err == nil && offset+int64(n) == r.Size() {
		return n, io.EOF
	}
	return n, err
}

type shortReaderAt struct{ *bytes.Reader }

func (r shortReaderAt) ReadAt(buf []byte, offset int64) (int, error) {
	n, _ := r.Reader.ReadAt(buf[:len(buf)-1], offset)
	return n, nil
}

// The helpers below name the index wire offsets used by corruption tests so
// those tests can describe the invariant they break instead of repeating
// byte-slice arithmetic inline. They follow the index layout in C2SP
// upki-revocation v1.0.0 (<https://c2sp.org/upki-revocation@v1.0.0#the-index>):
// a 14-byte header (8-byte magic, u16 num_filenames, u32 num_log_ids), then
// num_filenames 32-byte filename slots, then the log directory of 42-byte
// entries, each a 32-byte log_id, a u64 entry-section offset and a u16 entry
// count, all big-endian.

// testIndexLogEntry returns the 42-byte log directory entry for logIndex in
// an index with filenameCount filename slots.
func testIndexLogEntry(data []byte, filenameCount, logIndex int) []byte {
	start := headerSize + filenameCount*filenameSize + logIndex*logDirEntrySize

	return data[start : start+logDirEntrySize]
}

// testIndexLogID returns the log_id, the first 32 bytes of the entry.
func testIndexLogID(data []byte, filenameCount, logIndex int) []byte {
	return testIndexLogEntry(data, filenameCount, logIndex)[:32]
}

// testIndexEntryOffset returns the entry-section offset, the u64 that
// follows the log_id at bytes 32 through 39 of the entry.
func testIndexEntryOffset(data []byte, filenameCount, logIndex int) uint64 {
	return binary.BigEndian.Uint64(testIndexLogEntry(data, filenameCount, logIndex)[32:40])
}

// setTestIndexEntryOffset overwrites the entry-section offset in place.
func setTestIndexEntryOffset(data []byte, filenameCount, logIndex int, offset uint64) {
	binary.BigEndian.PutUint64(testIndexLogEntry(data, filenameCount, logIndex)[32:40], offset)
}

// swapTestIndexLogEntries exchanges two whole log directory entries, which
// breaks the directory's sort order without changing any entry's contents.

func swapTestIndexLogEntries(data []byte, filenameCount, a, b int) {
	entryA := bytes.Clone(testIndexLogEntry(data, filenameCount, a))
	entryB := bytes.Clone(testIndexLogEntry(data, filenameCount, b))
	copy(testIndexLogEntry(data, filenameCount, a), entryB)
	copy(testIndexLogEntry(data, filenameCount, b), entryA)
}
