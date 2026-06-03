// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package zip

import (
	"cmp"
	"compress/gzip"
	"errors"
	"fmt"
	"io"
	"os"
	"slices"
)

// A sparseFile represents an archive as a sequence of non-zero byte spans
// (the LFH headers, the Central Directory, the EOCD records, and any
// non-zero compressed bodies) plus a total length. Bytes outside any span
// are implicitly zero. This is the storage format used for goldens under
// testdata/zip64/ (suffix .zsparse) and the in-memory shape produced by
// the writer-reproduction harness.
//
// On-disk layout (all little-endian):
//
//	uint64 size
//	uint32 numSpans
//	for each span:
//		uint64 offset
//		uint32 dataLen
//		dataLen bytes
//
// Spans are sorted by offset and non-overlapping.
type sparseFile struct {
	Size  int64
	Spans []sparseSpan
}

type sparseSpan struct {
	Offset int64
	Data   []byte
}

// ReadAt implements [io.ReaderAt] by serving the underlying spans and
// synthesizing zero bytes for any gap inside [0, Size).
func (f *sparseFile) ReadAt(p []byte, off int64) (int, error) {
	if off < 0 {
		return 0, errors.New("sparseFile: negative offset")
	}
	if off >= f.Size {
		return 0, io.EOF
	}
	end := min(off+int64(len(p)), f.Size)
	n := int(end - off)
	clear(p[:n])
	for _, s := range f.Spans {
		sEnd := s.Offset + int64(len(s.Data))
		if sEnd <= off || s.Offset >= end {
			continue
		}
		from := max(s.Offset, off)
		to := min(sEnd, end)
		copy(p[from-off:to-off], s.Data[from-s.Offset:to-s.Offset])
	}
	if n < len(p) {
		return n, io.EOF
	}
	return n, nil
}

// materializeTail returns the last keep bytes of the conceptual file as a
// plain byte slice, suitable for [parseCD].
func (f *sparseFile) materializeTail(keep int64) (data []byte, baseOff uint64) {
	if keep > f.Size {
		keep = f.Size
	}
	base := f.Size - keep
	buf := make([]byte, keep)
	f.ReadAt(buf, base)
	return buf, uint64(base)
}

const sparseChunk = 4096

// scanSparse stream-reads r and builds a sparseFile, treating any contiguous
// run of zero bytes (rounded to sparseChunk boundaries) as a gap. Adjacent
// non-zero chunks are coalesced into one span.
func scanSparse(r io.Reader) (*sparseFile, error) {
	f := &sparseFile{}
	var cur *sparseSpan
	buf := make([]byte, sparseChunk)
	for {
		n, err := io.ReadFull(r, buf)
		if n > 0 {
			chunk := buf[:n]
			if isAllZero(chunk) {
				if cur != nil {
					f.Spans = append(f.Spans, *cur)
					cur = nil
				}
			} else {
				if cur == nil {
					cur = &sparseSpan{Offset: f.Size}
				}
				cur.Data = append(cur.Data, chunk...)
			}
			f.Size += int64(n)
		}
		if err != nil {
			if err == io.EOF || err == io.ErrUnexpectedEOF {
				break
			}
			return nil, err
		}
	}
	if cur != nil {
		f.Spans = append(f.Spans, *cur)
	}
	return f, nil
}

// writeSparse serializes f to w in the on-disk format described on
// [sparseFile].
func writeSparse(w io.Writer, f *sparseFile) error {
	var hdr [12]byte
	le.PutUint64(hdr[:8], uint64(f.Size))
	le.PutUint32(hdr[8:12], uint32(len(f.Spans)))
	if _, err := w.Write(hdr[:]); err != nil {
		return err
	}
	for _, s := range f.Spans {
		var b [12]byte
		le.PutUint64(b[:8], uint64(s.Offset))
		le.PutUint32(b[8:12], uint32(len(s.Data)))
		if _, err := w.Write(b[:]); err != nil {
			return err
		}
		if _, err := w.Write(s.Data); err != nil {
			return err
		}
	}
	return nil
}

// readSparse parses the on-disk format from r.
func readSparse(r io.Reader) (*sparseFile, error) {
	var hdr [12]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		return nil, err
	}
	f := &sparseFile{
		Size: int64(le.Uint64(hdr[:8])),
	}
	n := le.Uint32(hdr[8:12])
	if n > 1<<20 {
		return nil, fmt.Errorf("sparseFile: implausible span count %d", n)
	}
	f.Spans = make([]sparseSpan, n)
	for i := range f.Spans {
		var b [12]byte
		if _, err := io.ReadFull(r, b[:]); err != nil {
			return nil, err
		}
		f.Spans[i].Offset = int64(le.Uint64(b[:8]))
		sz := le.Uint32(b[8:12])
		f.Spans[i].Data = make([]byte, sz)
		if _, err := io.ReadFull(r, f.Spans[i].Data); err != nil {
			return nil, err
		}
	}
	if !slices.IsSortedFunc(f.Spans, func(a, b sparseSpan) int {
		return cmp.Compare(a.Offset, b.Offset)
	}) {
		return nil, errors.New("sparseFile: spans not sorted")
	}
	return f, nil
}

// readSparseFile reads a sparse file from path. The file is expected to be
// gzip-compressed; the outer gzip wrap shrinks goldens that contain non-zero
// compressed bodies (e.g., the deflate-zeros entries) by 100x because
// deflate-of-zeros is highly repetitive. Small Store goldens benefit too:
// gzip's header overhead is ~30 bytes, well under the bytes saved on a 4 KB
// sparse representation.
func readSparseFile(path string) (*sparseFile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	zr, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}
	defer zr.Close()
	return readSparse(zr)
}

// isAllZero reports whether every byte in b is 0.
func isAllZero(b []byte) bool {
	for _, c := range b {
		if c != 0 {
			return false
		}
	}
	return true
}

// sparseBuffer accumulates writes into a [sparseFile], dropping any
// chunkSize-byte chunk that is all-zero. This makes capturing the result
// of pushing multi-GiB streams of zeros through the writer almost free —
// the only bytes that end up retained are the LFHs, the Central
// Directory, the EOCD records, and any non-zero compressed body.
type sparseBuffer struct {
	f   sparseFile
	cur *sparseSpan
}

func (t *sparseBuffer) Write(p []byte) (int, error) {
	n := len(p)
	for len(p) > 0 {
		k := len(p)
		if k > sparseChunk {
			k = sparseChunk
		}
		chunk := p[:k]
		if isAllZero(chunk) {
			t.cur = nil
		} else {
			if t.cur == nil {
				t.f.Spans = append(t.f.Spans, sparseSpan{Offset: t.f.Size})
				t.cur = &t.f.Spans[len(t.f.Spans)-1]
			}
			t.cur.Data = append(t.cur.Data, chunk...)
		}
		t.f.Size += int64(k)
		p = p[k:]
	}
	return n, nil
}
