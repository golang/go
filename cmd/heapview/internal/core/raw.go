// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package core provides functions for reading core dumps
// and examining their contained heaps.
package core

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"runtime"
	"sort"
)

// RawDump provides raw access to the heap records in a core file.
// The raw records in this file are described by other structs named Raw{*}.
// All []byte slices are direct references to the underlying mmap'd file.
// These references will become invalid as soon as the RawDump is closed.
type RawDump struct {
	Params   *RawParams
	MemStats *runtime.MemStats

	HeapObjects    []RawSegment // heap objects sorted by Addr, low-to-high
	GlobalSegments []RawSegment // data, bss, and noptrbss segments

	OSThreads   []*RawOSThread
	Goroutines  []*RawGoroutine
	StackFrames []*RawStackFrame
	OtherRoots  []*RawOtherRoot
	Finalizers  []*RawFinalizer
	Defers      []*RawDefer
	Panics      []*RawPanic

	TypeFromItab map[uint64]uint64   // map from itab address to the type address that itab represents
	TypeFromAddr map[uint64]*RawType // map from RawType.Addr to RawType

	MemProfMap   map[uint64]*RawMemProfEntry
	AllocSamples []*RawAllocSample

	fmap *mmapFile
}

// RawParams holds metadata about the program that generated the dump.
type RawParams struct {
	// Info about the memory space

	ByteOrder binary.ByteOrder // byte order of all memory in this dump
	PtrSize   uint64           // in bytes
	HeapStart uint64           // heap start address
	HeapEnd   uint64           // heap end address (this is the last byte in the heap + 1)

	// Info about the program that generated this heapdump

	GoArch       string // GOARCH of the runtime library that generated this dump
	GoExperiment string // GOEXPERIMENT of the toolchain that build the runtime library
	NCPU         uint64 // number of physical cpus available to the program
}

// RawSegment represents a segment of memory.
type RawSegment struct {
	Addr      uint64       // base address of the segment
	Data      []byte       // data for this segment
	PtrFields RawPtrFields // offsets of ptr fields within this segment
}

// RawPtrFields represents a pointer field.
type RawPtrFields struct {
	encoded          []byte // list of uvarint-encoded offsets, or nil if none
	startOff, endOff uint64 // decoded offsets are translated and clipped to [startOff,endOff)
}

// RawOSThread represents an OS thread.
type RawOSThread struct {
	MAddr  uint64 // address of the OS thread descriptor (M)
	GoID   uint64 // go's internal ID for the thread
	ProcID uint64 // kernel's ID for the thread
}

// RawGoroutine represents a goroutine structure.
type RawGoroutine struct {
	GAddr        uint64 // address of the goroutine descriptor
	SP           uint64 // current stack pointer (lowest address in the currently running frame)
	GoID         uint64 // goroutine ID
	GoPC         uint64 // PC of the go statement that created this goroutine
	Status       uint64
	IsSystem     bool   // true if started by the system
	IsBackground bool   // always false in go1.7
	WaitSince    uint64 // time the goroutine started waiting, in nanoseconds since the Unix epoch
	WaitReason   string
	CtxtAddr     uint64 // address of the scheduling ctxt
	MAddr        uint64 // address of the OS thread descriptor (M)
	TopDeferAddr uint64 // address of the top defer record
	TopPanicAddr uint64 // address of the top panic record
}

// RawStackFrame represents a stack frame.
type RawStackFrame struct {
	Name     string
	Depth    uint64     // 0 = bottom of stack (currently running frame)
	CalleeSP uint64     // stack pointer of the child frame (or 0 for the bottom-most frame)
	EntryPC  uint64     // entry PC for the function
	PC       uint64     // current PC being executed
	NextPC   uint64     // for callers, where the function resumes (if anywhere) after the callee is done
	Segment  RawSegment // local vars (Segment.Addr is the stack pointer, i.e., lowest address in the frame)
}

// RawOtherRoot represents the other roots not in RawDump's other fields.
type RawOtherRoot struct {
	Description string
	Addr        uint64 // address pointed to by this root
}

// RawFinalizer represents a finalizer.
type RawFinalizer struct {
	IsQueued      bool   // if true, the object is unreachable and the finalizer is ready to run
	ObjAddr       uint64 // address of the object to finalize
	ObjTypeAddr   uint64 // address of the descriptor for typeof(obj)
	FnAddr        uint64 // function to be run (a FuncVal*)
	FnArgTypeAddr uint64 // address of the descriptor for the type of the function argument
	FnPC          uint64 // PC of finalizer entry point
}

// RawDefer represents a defer.
type RawDefer struct {
	Addr     uint64 // address of the defer record
	GAddr    uint64 // address of the containing goroutine's descriptor
	ArgP     uint64 // stack pointer giving the args for defer (TODO: is this right?)
	PC       uint64 // PC of the defer instruction
	FnAddr   uint64 // function to be run (a FuncVal*)
	FnPC     uint64 // PC of the defered function's entry point
	LinkAddr uint64 // address of the next defer record in this chain
}

// RawPanic represents a panic.
type RawPanic struct {
	Addr        uint64 // address of the panic record
	GAddr       uint64 // address of the containing goroutine's descriptor
	ArgTypeAddr uint64 // type of the panic arg
	ArgAddr     uint64 // address of the panic arg
	DeferAddr   uint64 // address of the defer record that is currently running
	LinkAddr    uint64 // address of the next panic record in this chain
}

// RawType repesents the Go runtime's representation of a type.
type RawType struct {
	Addr uint64 // address of the type descriptor
	Size uint64 // in bytes
	Name string // not necessarily unique
	// If true, this type is equivalent to a single pointer, so ifaces can store
	// this type directly in the data field (without indirection).
	DirectIFace bool
}

// RawMemProfEntry represents a memory profiler entry.
type RawMemProfEntry struct {
	Size      uint64            // size of the allocated object
	NumAllocs uint64            // number of allocations
	NumFrees  uint64            // number of frees
	Stacks    []RawMemProfFrame // call stacks
}

// RawMemProfFrame represents a memory profiler frame.
type RawMemProfFrame struct {
	Func []byte // string left as []byte reference to save memory
	File []byte // string left as []byte reference to save memory
	Line uint64
}

// RawAllocSample represents a memory profiler allocation sample.
type RawAllocSample struct {
	Addr uint64           // address of object
	Prof *RawMemProfEntry // record of allocation site
}

// Close closes the file.
func (r *RawDump) Close() error {
	return r.fmap.Close()
}

// FindSegment returns the segment that contains the given address, or
// nil of no segment contains the address.
func (r *RawDump) FindSegment(addr uint64) *RawSegment {
	// Binary search for an upper-bound heap object, then check
	// if the previous object contains addr.
	k := sort.Search(len(r.HeapObjects), func(k int) bool {
		return addr < r.HeapObjects[k].Addr
	})
	k--
	if k >= 0 && r.HeapObjects[k].Contains(addr) {
		return &r.HeapObjects[k]
	}

	// Check all global segments.
	for k := range r.GlobalSegments {
		if r.GlobalSegments[k].Contains(addr) {
			return &r.GlobalSegments[k]
		}
	}

	// NB: Stack-local vars are technically allocated in the heap, since stack frames are
	// allocated in the heap space, however, stack frames don't show up in r.HeapObjects.
	for _, f := range r.StackFrames {
		if f.Segment.Contains(addr) {
			return &f.Segment
		}
	}

	return nil
}

// Contains returns true if the segment contains the given address.
func (r RawSegment) Contains(addr uint64) bool {
	return r.Addr <= addr && addr < r.Addr+r.Size()
}

// ContainsRange returns true if the segment contains the range [addr, addr+size).
func (r RawSegment) ContainsRange(addr, size uint64) bool {
	if !r.Contains(addr) {
		return false
	}
	if size > 0 && !r.Contains(addr+size-1) {
		return false
	}
	return true
}

// Size returns the size of the segment in bytes.
func (r RawSegment) Size() uint64 {
	return uint64(len(r.Data))
}

// Slice takes a slice of the given segment. Panics if [offset,offset+size)
// is out-of-bounds. The resulting RawSegment.PtrOffsets will clipped and
// translated into the new segment.
func (r RawSegment) Slice(offset, size uint64) *RawSegment {
	if offset+size > uint64(len(r.Data)) {
		panic(fmt.Errorf("slice(%d,%d) out-of-bounds of segment @%x sz=%d", offset, size, r.Addr, len(r.Data)))
	}
	return &RawSegment{
		Addr: r.Addr + offset,
		Data: r.Data[offset : offset+size : offset+size],
		PtrFields: RawPtrFields{
			encoded:  r.PtrFields.encoded,
			startOff: r.PtrFields.startOff + offset,
			endOff:   r.PtrFields.startOff + offset + size,
		},
	}
}

// Offsets decodes the list of ptr field offsets.
func (r RawPtrFields) Offsets() []uint64 {
	if r.encoded == nil {
		return nil
	}

	// NB: This should never fail since we already decoded the varints once
	// when parsing the file originally. Hence we panic on failure.
	reader := bytes.NewReader(r.encoded)
	readUint64 := func() uint64 {
		x, err := binary.ReadUvarint(reader)
		if err != nil {
			panic(fmt.Errorf("unexpected failure decoding uvarint: %v", err))
		}
		return x
	}

	var out []uint64
	for {
		k := readUint64()
		switch k {
		case 0: // end
			return out
		case 1: // ptr
			x := readUint64()
			if r.startOff <= x && x < r.endOff {
				out = append(out, x-r.startOff)
			}
		default:
			panic(fmt.Errorf("unexpected FieldKind %d", k))
		}
	}
}

// ReadPtr decodes a ptr from the given byte slice.
func (r *RawParams) ReadPtr(b []byte) uint64 {
	switch r.PtrSize {
	case 4:
		return uint64(r.ByteOrder.Uint32(b))
	case 8:
		return r.ByteOrder.Uint64(b)
	default:
		panic(fmt.Errorf("unsupported PtrSize=%d", r.PtrSize))
	}
}

// WritePtr encodes a ptr into the given byte slice.
func (r *RawParams) WritePtr(b []byte, addr uint64) {
	switch r.PtrSize {
	case 4:
		r.ByteOrder.PutUint32(b, uint32(addr))
	case 8:
		r.ByteOrder.PutUint64(b, addr)
	default:
		panic(fmt.Errorf("unsupported PtrSize=%d", r.PtrSize))
	}
}
