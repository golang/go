// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/internal/sys"
	"cmd/link/internal/loader"
	"encoding/binary"
	"errors"
	"log"
	"os"
)

// If fallocate is not supported on this platform, return this error. The error
// is ignored where needed, and OutBuf writes to heap memory.
var errNoFallocate = errors.New("operation not supported")

const outbufMode = 0775

// OutBuf is a buffered file writer.
//
// It is similar to the Writer in cmd/internal/bio with a few small differences.
//
// First, it tracks the output architecture and uses it to provide
// endian helpers.
//
// Second, it provides a very cheap offset counter that doesn't require
// any system calls to read the value.
//
// Third, it also mmaps the output file (if available). The intended usage is:
//   - Mmap the output file
//   - Write the content
//   - possibly apply any edits in the output buffer
//   - possibly write more content to the file. These writes take place in a heap
//     backed buffer that will get synced to disk.
//   - Munmap the output file
//
// And finally, it provides a mechanism by which you can multithread the
// writing of output files. This mechanism is accomplished by copying a OutBuf,
// and using it in the thread/goroutine.
//
// Parallel OutBuf is intended to be used like:
//
//	func write(out *OutBuf) {
//	  var wg sync.WaitGroup
//	  for i := 0; i < 10; i++ {
//	    wg.Add(1)
//	    view, err := out.View(start[i])
//	    if err != nil {
//	       // handle output
//	       continue
//	    }
//	    go func(out *OutBuf, i int) {
//	      // do output
//	      wg.Done()
//	    }(view, i)
//	  }
//	  wg.Wait()
//	}
type OutBuf struct {
	arch *sys.Arch
	off  int64

	buf  []byte // backing store of mmap'd output file
	heap []byte // backing store for non-mmapped data

	name   string
	f      *os.File
	encbuf [8]byte // temp buffer used by WriteN methods
	isView bool    // true if created from View()
}

func (out *OutBuf) Open(name string) error {
	if out.f != nil {
		return errors.New("cannot open more than one file")
	}
	f, err := os.OpenFile(name, os.O_RDWR|os.O_CREATE|os.O_TRUNC, outbufMode)
	if err != nil {
		return err
	}
	out.off = 0
	out.name = name
	out.f = f
	return nil
}

func NewOutBuf(arch *sys.Arch) *OutBuf {
	return &OutBuf{
		arch: arch,
	}
}

func (out *OutBuf) View(start uint64) *OutBuf {
	return &OutBuf{
		arch:   out.arch,
		name:   out.name,
		buf:    out.buf,
		heap:   out.heap,
		off:    int64(start),
		isView: true,
	}
}

var viewCloseError = errors.New("cannot Close OutBuf from View")

func (out *OutBuf) Close() error {
	if out.isView {
		return viewCloseError
	}
	if out.isMmapped() {
		out.copyHeap()
		out.purgeSignatureCache()
		out.munmap()
	}
	if out.f == nil {
		return nil
	}
	if len(out.heap) != 0 {
		if _, err := out.f.Write(out.heap); err != nil {
			return err
		}
	}
	if err := out.f.Close(); err != nil {
		return err
	}
	out.f = nil
	return nil
}

// ErrorClose closes the output file (if any).
// It is supposed to be called only at exit on error, so it doesn't do
// any clean up or buffer flushing, just closes the file.
func (out *OutBuf) ErrorClose() {
	if out.isView {
		panic(viewCloseError)
	}
	if out.f == nil {
		return
	}
	out.f.Close() // best effort, ignore error
	out.f = nil
}

// isMmapped returns true if the OutBuf is mmaped.
func (out *OutBuf) isMmapped() bool {
	return len(out.buf) != 0
}

// Data returns the whole written OutBuf as a byte slice.
func (out *OutBuf) Data() []byte {
	if out.isMmapped() {
		out.copyHeap()
		return out.buf
	}
	return out.heap
}

// copyHeap copies the heap to the mmapped section of memory, returning true if
// a copy takes place.
func (out *OutBuf) copyHeap() bool {
	if !out.isMmapped() { // only valuable for mmapped OutBufs.
		return false
	}
	if out.isView {
		panic("can't copyHeap a view")
	}

	bufLen := len(out.buf)
	heapLen := len(out.heap)
	total := uint64(bufLen + heapLen)
	if heapLen != 0 {
		if err := out.Mmap(total); err != nil { // Mmap will copy out.heap over to out.buf
			Exitf("mapping output file failed: %v", err)
		}
	}
	return true
}

// maxOutBufHeapLen limits the growth of the heap area.
const maxOutBufHeapLen = 10 << 20

// writeLoc determines the write location if a buffer is mmaped.
// We maintain two write buffers, an mmapped section, and a heap section for
// writing. When the mmapped section is full, we switch over the heap memory
// for writing.
func (out *OutBuf) writeLoc(lenToWrite int64) (int64, []byte) {
	// See if we have enough space in the mmaped area.
	bufLen := int64(len(out.buf))
	if out.off+lenToWrite <= bufLen {
		return out.off, out.buf
	}

	// Not enough space in the mmaped area, write to heap area instead.
	heapPos := out.off - bufLen
	heapLen := int64(len(out.heap))
	lenNeeded := heapPos + lenToWrite
	if lenNeeded > heapLen { // do we need to grow the heap storage?
		// The heap variables aren't protected by a mutex. For now, just bomb if you
		// try to use OutBuf in parallel. (Note this probably could be fixed.)
		if out.isView {
			panic("cannot write to heap in parallel")
		}
		// See if our heap would grow to be too large, and if so, copy it to the end
		// of the mmapped area.
		if heapLen > maxOutBufHeapLen && out.copyHeap() {
			heapPos -= heapLen
			lenNeeded = heapPos + lenToWrite
			heapLen = 0
		}
		out.heap = append(out.heap, make([]byte, lenNeeded-heapLen)...)
	}
	return heapPos, out.heap
}

func (out *OutBuf) SeekSet(p int64) {
	out.off = p
}

func (out *OutBuf) Offset() int64 {
	return out.off
}

// Write writes the contents of v to the buffer.
func (out *OutBuf) Write(v []byte) (int, error) {
	n := len(v)
	pos, buf := out.writeLoc(int64(n))
	copy(buf[pos:], v)
	out.off += int64(n)
	return n, nil
}

func (out *OutBuf) Write8(v uint8) {
	pos, buf := out.writeLoc(1)
	buf[pos] = v
	out.off++
}

// WriteByte is an alias for Write8 to fulfill the io.ByteWriter interface.
func (out *OutBuf) WriteByte(v byte) error {
	out.Write8(v)
	return nil
}

func (out *OutBuf) Write16(v uint16) {
	out.arch.ByteOrder.PutUint16(out.encbuf[:], v)
	out.Write(out.encbuf[:2])
}

func (out *OutBuf) Write32(v uint32) {
	out.arch.ByteOrder.PutUint32(out.encbuf[:], v)
	out.Write(out.encbuf[:4])
}

func (out *OutBuf) Write32b(v uint32) {
	binary.BigEndian.PutUint32(out.encbuf[:], v)
	out.Write(out.encbuf[:4])
}

func (out *OutBuf) Write64(v uint64) {
	out.arch.ByteOrder.PutUint64(out.encbuf[:], v)
	out.Write(out.encbuf[:8])
}

func (out *OutBuf) Write64b(v uint64) {
	binary.BigEndian.PutUint64(out.encbuf[:], v)
	out.Write(out.encbuf[:8])
}

func (out *OutBuf) WriteString(s string) {
	pos, buf := out.writeLoc(int64(len(s)))
	n := copy(buf[pos:], s)
	if n != len(s) {
		log.Fatalf("WriteString truncated. buffer size: %d, offset: %d, len(s)=%d", len(out.buf), out.off, len(s))
	}
	out.off += int64(n)
}

// WriteStringN writes the first n bytes of s.
// If n is larger than len(s) then it is padded with zero bytes.
func (out *OutBuf) WriteStringN(s string, n int) {
	out.WriteStringPad(s, n, zeros[:])
}

// WriteStringPad writes the first n bytes of s.
// If n is larger than len(s) then it is padded with the bytes in pad (repeated as needed).
func (out *OutBuf) WriteStringPad(s string, n int, pad []byte) {
	if len(s) >= n {
		out.WriteString(s[:n])
	} else {
		out.WriteString(s)
		n -= len(s)
		for n > len(pad) {
			out.Write(pad)
			n -= len(pad)

		}
		out.Write(pad[:n])
	}
}

// WriteSym writes the content of a Symbol, and returns the output buffer
// that we just wrote, so we can apply further edit to the symbol content.
// For generator symbols, it also sets the symbol's Data to the output
// buffer.
func (out *OutBuf) WriteSym(ldr *loader.Loader, s loader.Sym) []byte {
	if !ldr.IsGeneratedSym(s) {
		P := ldr.Data(s)
		n := int64(len(P))
		pos, buf := out.writeLoc(n)
		copy(buf[pos:], P)
		out.off += n
		ldr.FreeData(s)
		return buf[pos : pos+n]
	} else {
		n := ldr.SymSize(s)
		pos, buf := out.writeLoc(n)
		out.off += n
		ldr.MakeSymbolUpdater(s).SetData(buf[pos : pos+n])
		return buf[pos : pos+n]
	}
}
