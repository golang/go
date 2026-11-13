// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package upki

import (
	"bytes"
	"cmp"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"slices"
	"syscall"
)

// On-disk layout of index.bin from
// https://c2sp.org/upki-revocation@v1.0.0.
//
//	HEADER (14 bytes):
//	  magic:         [8]byte    "upkiidx1"
//	  num_filenames: u16 BE
//	  num_log_ids:   u32 BE
//
//	TABLES (read eagerly):
//	  filename[num_filenames]:   [32]byte  restricted ASCII, NULL-padded
//	  log_dir[num_log_ids]:                sorted lexicographically by log_id
//	    log_id:      [32]byte
//	    offset:      u64 BE                file offset of entry section
//	    num_entries: u16 BE
//
//	ENTRY SECTIONS (read on demand via ReadAt):
//	  entry[num_entries]:
//	    filter_index:  u16 BE
//	    min_timestamp: u64 BE
//	    max_timestamp: u64 BE
//
// The legacy "upkiidx0" format encoded num_filenames and filter_index as u8;
// it is not supported.
const (
	IndexFilename = "index.bin"
	indexMagic    = "upkiidx1"
	filenameSize  = 32 // NULL-padded filename slot, restricted ASCII

	headerSize      = 8 + 2 + 4  // 8 byte magic + num_filenames(u16) + num_log_ids(u32)
	logDirEntrySize = 32 + 8 + 2 // 32 byte log_id + entry-section offset(u64) + count(u16)
	entrySize       = 2 + 8 + 8  // filter_idx(u16) + min_ts(u64) + max_ts(u64)
)

var (
	// ErrNoData reports that the cache index is absent or inaccessible.
	ErrNoData = errors.New("upki: no revocation data")

	errInvalidIndex = errors.New("upki: invalid index")
)

// OpenIndex opens the UPKI index at path. If the path is missing, a parent
// is not a directory, or access is denied, the error wraps ErrNoData.
func OpenIndex(path string) (*os.File, error) {
	f, err := openCacheFile(path)
	if errors.Is(err, os.ErrNotExist) || errors.Is(err, syscall.ENOTDIR) || errors.Is(err, os.ErrPermission) {
		return nil, fmt.Errorf("%w: %w", ErrNoData, err)
	}
	return f, err
}

// index is the in-memory index of revocation filter coverage built from an
// upki revocation index.bin.
//
// Its purpose is to avoid needing to read each of the cached clubcard
// filters into memory for each revocation check.
//
// Constructing an index reads the fixed header, the filename table, and
// the log-id directory eagerly. Per-log entry sections are read on demand
// via [io.ReaderAt.ReadAt] during [index.lookup].
//
// index is safe for concurrent [index.lookup] calls: after construction
// only the [io.ReaderAt] is used, and [io.ReaderAt.ReadAt] does not share
// state with itself.
type index struct {
	filenames []string
	logDir    []byte // packed: numLogs * logDirEntrySize
	numLogs   int
	r         *os.File
}

// newIndex eagerly reads the header and lookup tables from f. Entry sections
// are read on demand during [index.lookup].
func newIndex(f *os.File) (*index, error) {
	// header is: magic[8] | num_filenames(u16 BE) | num_log_ids(u32 BE).
	var header [headerSize]byte
	if _, err := readFullAt(f, header[:], 0); err != nil {
		return nil, fmt.Errorf("%w: read header: %w", errInvalidIndex, err)
	}
	if !bytes.Equal(header[:8], []byte(indexMagic)) {
		return nil, fmt.Errorf("%w: bad magic", errInvalidIndex)
	}
	numFilenames := int(binary.BigEndian.Uint16(header[8:10]))
	numLogIDs := binary.BigEndian.Uint32(header[10:14])

	// The filename table and log directory are contiguous and their sizes are
	// fully determined by the header. Compute their size before converting
	// num_log_ids to int so a large value cannot wrap on 32-bit platforms.
	filenamesLen := numFilenames * filenameSize
	tablesLen := int64(filenamesLen) + int64(numLogIDs)*logDirEntrySize
	info, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("%w: stat index: %w", errInvalidIndex, err)
	}
	if int64(headerSize)+tablesLen > info.Size() || tablesLen > int64(^uint(0)>>1) {
		return nil, fmt.Errorf("%w: read tables: %w", errInvalidIndex, io.ErrUnexpectedEOF)
	}
	numLogs := int(numLogIDs)
	tables := make([]byte, int(tablesLen))
	if _, err := readFullAt(f, tables, headerSize); err != nil {
		return nil, fmt.Errorf("%w: read tables: %w", errInvalidIndex, err)
	}
	filenamesBuf := tables[:filenamesLen]
	logDir := tables[filenamesLen:]

	filenames := make([]string, numFilenames)
	for i := range numFilenames {
		slot := filenamesBuf[i*filenameSize : (i+1)*filenameSize]
		end := bytes.IndexByte(slot, 0)
		if end < 0 {
			end = filenameSize
		}

		name := string(slot[:end])
		if err := validateFilterFilename(name); err != nil {
			return nil, fmt.Errorf("%w: filename table entry %d: %w", errInvalidIndex, i, err)
		}

		filenames[i] = name
	}

	if err := validateLogDirOrder(logDir, numLogs); err != nil {
		return nil, err
	}
	tablesEnd := int64(headerSize) + tablesLen
	if err := validateEntrySections(logDir, numLogs, tablesEnd, info.Size()); err != nil {
		return nil, err
	}

	return &index{
		filenames: filenames,
		logDir:    logDir,
		numLogs:   numLogs,
		r:         f,
	}, nil
}

// validateFilterFilename enforces the spec's filter filename restrictions:
// 1 to 32 bytes, only ASCII `A`-`Z`, `a`-`z`, `0`-`9`, `-`, `.`, and `_`,
// and never the names "." or "..".
//
// Filenames from the index are used to open filter files relative to the
// cache directory, so this must reject anything that could escape it (path
// separators, "..", NUL is already excluded by slot parsing) before any
// name reaches a filesystem operation. An index containing a violating
// name is malformed.
func validateFilterFilename(name string) error {
	if len(name) == 0 {
		return errors.New("empty filename")
	}
	if name == "." || name == ".." {
		return fmt.Errorf("filename %q is a directory reference", name)
	}
	for i := 0; i < len(name); i++ {
		switch c := name[i]; {
		case c >= 'A' && c <= 'Z', c >= 'a' && c <= 'z', c >= '0' && c <= '9',
			c == '-', c == '.', c == '_':
		default:
			return fmt.Errorf("filename contains disallowed byte 0x%02x", c)
		}
	}

	return nil
}

// validateLogDirOrder checks that the log directory is strictly ascending by
// log_id.
//
// The spec requires entries to be sorted lexicographically with no duplicate
// log_id. Checking for strict ordering enforces both at once.
//
// findLog's binary search depends on this order, so an unsorted directory would
// otherwise miss present logs (yielding a wrong not-covered result), while a
// duplicate would make lookups nondeterministic.
func validateLogDirOrder(logDir []byte, numLogs int) error {
	for i := 1; i < numLogs; i++ {
		prev := logDir[(i-1)*logDirEntrySize : (i-1)*logDirEntrySize+32]
		cur := logDir[i*logDirEntrySize : i*logDirEntrySize+32]
		if bytes.Compare(prev, cur) >= 0 {
			return fmt.Errorf("%w: log directory not strictly sorted by log id at index %d",
				errInvalidIndex, i)
		}
	}

	return nil
}

// validateEntrySections checks every log directory entry's on-demand entry
// section against the spec's bounds: it MUST lie entirely within the file,
// MUST NOT overlap the header, filename table, or log directory, and SHOULD
// NOT overlap another log's entry section.
func validateEntrySections(logDir []byte, numLogs int, tablesEnd, fileSize int64) error {
	type span struct{ start, end int64 }

	spans := make([]span, 0, numLogs)
	for i := range numLogs {
		off := i * logDirEntrySize
		entryOffset := int64(binary.BigEndian.Uint64(logDir[off+32 : off+40]))
		count := int64(binary.BigEndian.Uint16(logDir[off+40 : off+42]))
		sectionLen := count * entrySize

		if entryOffset < tablesEnd {
			return fmt.Errorf("%w: entry section offset %d overlaps header/tables (end %d)",
				errInvalidIndex, entryOffset, tablesEnd)
		}
		if entryOffset > fileSize || sectionLen > fileSize-entryOffset {
			return fmt.Errorf("%w: entry section at offset %d with length %d exceeds file size %d",
				errInvalidIndex, entryOffset, sectionLen, fileSize)
		}

		if sectionLen > 0 {
			spans = append(spans, span{start: entryOffset, end: entryOffset + sectionLen})
		}
	}

	// The spec leaves overlap between distinct logs' entry sections
	// unspecified and says implementations SHOULD reject it; aliased
	// sections would make one log's entries silently double as another's.
	slices.SortFunc(spans, func(a, b span) int { return cmp.Compare(a.start, b.start) })
	for i := 1; i < len(spans); i++ {
		if spans[i].start < spans[i-1].end {
			return fmt.Errorf("%w: entry sections [%d, %d) and [%d, %d) overlap",
				errInvalidIndex, spans[i-1].start, spans[i-1].end, spans[i].start, spans[i].end)
		}
	}

	return nil
}

// lookup finds the filter files that cover the given CT log id at the
// given timestamp, returning the covering filters' basenames in index
// entry order.
//
// It binary-searches the in-memory log-id directory and on a hit issues
// a single [io.ReaderAt.ReadAt] against that log's entry section. Every
// entry whose interval [min, max] contains timestamp contributes its
// filter. A log can have several covering filters for the same instant
// and a conclusive revocation answer may come from any of them, so
// callers must consult all the returned filters, not just the first.
//
// The result is empty (with a nil error) when no entry covers the
// (log id, timestamp). In this case the log is not indexed, or it is
// indexed but the timestamp falls outside every recorded interval.
func (idx *index) lookup(id logID, ts timestamp) ([]string, error) {
	dirOffset, ok := idx.findLog(id)
	if !ok {
		return nil, nil
	}

	// log-dir entry is: log_id[32] | entry_offset(u64) | count(u16).
	entryOffset := binary.BigEndian.Uint64(idx.logDir[dirOffset+32 : dirOffset+40])
	count := binary.BigEndian.Uint16(idx.logDir[dirOffset+40 : dirOffset+42])

	buf := make([]byte, int(count)*entrySize)
	if _, err := readFullAt(idx.r, buf, int64(entryOffset)); err != nil {
		return nil, fmt.Errorf("%w: read entries: %w", errInvalidIndex, err)
	}

	var filenames []string
	for i := range int(count) {
		// entry is: filter_idx(u16) | min_ts(u64) | max_ts(u64).
		off := i * entrySize
		filterIdx := int(binary.BigEndian.Uint16(buf[off : off+2]))
		if filterIdx >= len(idx.filenames) {
			return nil, fmt.Errorf("%w: entry filter index %d out of range", errInvalidIndex, filterIdx)
		}
		minTS := binary.BigEndian.Uint64(buf[off+2 : off+10])
		maxTS := binary.BigEndian.Uint64(buf[off+10 : off+18])
		if minTS > uint64(ts) || uint64(ts) > maxTS {
			continue
		}

		filenames = append(filenames, idx.filenames[filterIdx])
	}

	return filenames, nil
}

// findLog binary-searches the lexicographically-sorted log-id directory
// for logID and returns the byte offset (within idx.logDir) of the
// matching entry, or (0, false) if absent.
//
// Hand-rolled rather than sort.Search / slices.BinarySearchFunc because
// the directory is a packed byte buffer with no element type, and we don't
// want to transform to a slice.
func (idx *index) findLog(id logID) (int, bool) {
	lo, hi := 0, idx.numLogs
	target := id[:]
	for lo < hi {
		mid := int(uint(lo+hi) >> 1)
		off := mid * logDirEntrySize
		cmp := bytes.Compare(idx.logDir[off:off+32], target)
		switch {
		case cmp < 0:
			lo = mid + 1
		case cmp > 0:
			hi = mid
		default:
			return off, true
		}
	}

	return 0, false
}

func readFullAt(r io.ReaderAt, buf []byte, off int64) (int, error) {
	n, err := r.ReadAt(buf, off)
	if err == io.EOF && n == len(buf) {
		return n, nil
	}
	if err == nil && n < len(buf) {
		return n, io.ErrUnexpectedEOF
	}

	return n, err
}
