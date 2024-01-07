// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package zstd provides a decompressor for zstd streams,
// described in RFC 8878. It does not support dictionaries.
package zstd

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
)

// fuzzing is a fuzzer hook set to true when fuzzing.
// This is used to reject cases where we don't match zstd.
var fuzzing = false

// Reader implements [io.Reader] to read a zstd compressed stream.
type Reader struct {
	// The underlying Reader.
	r io.Reader

	// Whether we have read the frame header.
	// This is of interest when buffer is empty.
	// If true we expect to see a new block.
	sawFrameHeader bool

	// Whether the current frame expects a checksum.
	hasChecksum bool

	// Whether we have read at least one frame.
	readOneFrame bool

	// True if the frame size is not known.
	frameSizeUnknown bool

	// The number of uncompressed bytes remaining in the current frame.
	// If frameSizeUnknown is true, this is not valid.
	remainingFrameSize uint64

	// The number of bytes read from r up to the start of the current
	// block, for error reporting.
	blockOffset int64

	// Buffered decompressed data.
	buffer []byte
	// Current read offset in buffer.
	off int

	// The current repeated offsets.
	repeatedOffset1 uint32
	repeatedOffset2 uint32
	repeatedOffset3 uint32

	// The current Huffman tree used for compressing literals.
	huffmanTable     []uint16
	huffmanTableBits int

	// The window for back references.
	window window

	// A buffer available to hold a compressed block.
	compressedBuf []byte

	// A buffer for literals.
	literals []byte

	// Sequence decode FSE tables.
	seqTables    [3][]fseBaselineEntry
	seqTableBits [3]uint8

	// Buffers for sequence decode FSE tables.
	seqTableBuffers [3][]fseBaselineEntry

	// Scratch space used for small reads, to avoid allocation.
	scratch [16]byte

	// A scratch table for reading an FSE. Only temporarily valid.
	fseScratch []fseEntry

	// For checksum computation.
	checksum xxhash64
}

// NewReader creates a new Reader that decompresses data from the given reader.
func NewReader(input io.Reader) *Reader {
	r := new(Reader)
	r.Reset(input)
	return r
}

// Reset discards the current state and starts reading a new stream from r.
// This permits reusing a Reader rather than allocating a new one.
func (r *Reader) Reset(input io.Reader) {
	r.r = input

	// Several fields are preserved to avoid allocation.
	// Others are always set before they are used.
	r.sawFrameHeader = false
	r.hasChecksum = false
	r.readOneFrame = false
	r.frameSizeUnknown = false
	r.remainingFrameSize = 0
	r.blockOffset = 0
	r.buffer = r.buffer[:0]
	r.off = 0
	// repeatedOffset1
	// repeatedOffset2
	// repeatedOffset3
	// huffmanTable
	// huffmanTableBits
	// window
	// compressedBuf
	// literals
	// seqTables
	// seqTableBits
	// seqTableBuffers
	// scratch
	// fseScratch
}

// Read implements [io.Reader].
func (r *Reader) Read(p []byte) (int, error) {
	if err := r.refillIfNeeded(); err != nil {
		return 0, err
	}
	n := copy(p, r.buffer[r.off:])
	r.off += n
	return n, nil
}

// ReadByte implements [io.ByteReader].
func (r *Reader) ReadByte() (byte, error) {
	if err := r.refillIfNeeded(); err != nil {
		return 0, err
	}
	ret := r.buffer[r.off]
	r.off++
	return ret, nil
}

// refillIfNeeded reads the next block if necessary.
func (r *Reader) refillIfNeeded() error {
	for r.off >= len(r.buffer) {
		if err := r.refill(); err != nil {
			return err
		}
		r.off = 0
	}
	return nil
}

// refill reads and decompresses the next block.
func (r *Reader) refill() error {
	if !r.sawFrameHeader {
		if err := r.readFrameHeader(); err != nil {
			return err
		}
	}
	return r.readBlock()
}

// readFrameHeader reads the frame header and prepares to read a block.
func (r *Reader) readFrameHeader() error {
retry:
	relativeOffset := 0

	// Read magic number. RFC 3.1.1.
	if _, err := io.ReadFull(r.r, r.scratch[:4]); err != nil {
		// We require that the stream contains at least one frame.
		if err == io.EOF && !r.readOneFrame {
			err = io.ErrUnexpectedEOF
		}
		return r.wrapError(relativeOffset, err)
	}

	if magic := binary.LittleEndian.Uint32(r.scratch[:4]); magic != 0xfd2fb528 {
		if magic >= 0x184d2a50 && magic <= 0x184d2a5f {
			// This is a skippable frame.
			r.blockOffset += int64(relativeOffset) + 4
			if err := r.skipFrame(); err != nil {
				return err
			}
			r.readOneFrame = true
			goto retry
		}

		return r.makeError(relativeOffset, "invalid magic number")
	}

	relativeOffset += 4

	// Read Frame_Header_Descriptor. RFC 3.1.1.1.1.
	if _, err := io.ReadFull(r.r, r.scratch[:1]); err != nil {
		return r.wrapNonEOFError(relativeOffset, err)
	}
	descriptor := r.scratch[0]

	singleSegment := descriptor&(1<<5) != 0

	fcsFieldSize := 1 << (descriptor >> 6)
	if fcsFieldSize == 1 && !singleSegment {
		fcsFieldSize = 0
	}

	var windowDescriptorSize int
	if singleSegment {
		windowDescriptorSize = 0
	} else {
		windowDescriptorSize = 1
	}

	if descriptor&(1<<3) != 0 {
		return r.makeError(relativeOffset, "reserved bit set in frame header descriptor")
	}

	r.hasChecksum = descriptor&(1<<2) != 0
	if r.hasChecksum {
		r.checksum.reset()
	}

	// Dictionary_ID_Flag. RFC 3.1.1.1.1.6.
	dictionaryIdSize := 0
	if dictIdFlag := descriptor & 3; dictIdFlag != 0 {
		dictionaryIdSize = 1 << (dictIdFlag - 1)
	}

	relativeOffset++

	headerSize := windowDescriptorSize + dictionaryIdSize + fcsFieldSize

	if _, err := io.ReadFull(r.r, r.scratch[:headerSize]); err != nil {
		return r.wrapNonEOFError(relativeOffset, err)
	}

	// Figure out the maximum amount of data we need to retain
	// for backreferences.
	var windowSize int
	if !singleSegment {
		// Window descriptor. RFC 3.1.1.1.2.
		windowDescriptor := r.scratch[0]
		exponent := uint64(windowDescriptor >> 3)
		mantissa := uint64(windowDescriptor & 7)
		windowLog := exponent + 10
		windowBase := uint64(1) << windowLog
		windowAdd := (windowBase / 8) * mantissa
		windowSize = int(windowBase + windowAdd)

		// Default zstd sets limits on the window size.
		if fuzzing && (windowLog > 31 || windowSize > 1<<27) {
			return r.makeError(relativeOffset, "windowSize too large")
		}
	}

	// Dictionary_ID. RFC 3.1.1.1.3.
	if dictionaryIdSize != 0 {
		dictionaryId := r.scratch[windowDescriptorSize : windowDescriptorSize+dictionaryIdSize]
		// Allow only zero Dictionary ID.
		for _, b := range dictionaryId {
			if b != 0 {
				return r.makeError(relativeOffset, "dictionaries are not supported")
			}
		}
	}

	// Frame_Content_Size. RFC 3.1.1.1.4.
	r.frameSizeUnknown = false
	r.remainingFrameSize = 0
	fb := r.scratch[windowDescriptorSize+dictionaryIdSize:]
	switch fcsFieldSize {
	case 0:
		r.frameSizeUnknown = true
	case 1:
		r.remainingFrameSize = uint64(fb[0])
	case 2:
		r.remainingFrameSize = 256 + uint64(binary.LittleEndian.Uint16(fb))
	case 4:
		r.remainingFrameSize = uint64(binary.LittleEndian.Uint32(fb))
	case 8:
		r.remainingFrameSize = binary.LittleEndian.Uint64(fb)
	default:
		panic("unreachable")
	}

	// RFC 3.1.1.1.2.
	// When Single_Segment_Flag is set, Window_Descriptor is not present.
	// In this case, Window_Size is Frame_Content_Size.
	if singleSegment {
		windowSize = int(r.remainingFrameSize)
	}

	// RFC 8878 3.1.1.1.1.2. permits us to set an 8M max on window size.
	if windowSize > 8<<20 {
		windowSize = 8 << 20
	}

	relativeOffset += headerSize

	r.sawFrameHeader = true
	r.readOneFrame = true
	r.blockOffset += int64(relativeOffset)

	// Prepare to read blocks from the frame.
	r.repeatedOffset1 = 1
	r.repeatedOffset2 = 4
	r.repeatedOffset3 = 8
	r.huffmanTableBits = 0
	r.window.reset(windowSize)
	r.seqTables[0] = nil
	r.seqTables[1] = nil
	r.seqTables[2] = nil

	return nil
}

// skipFrame skips a skippable frame. RFC 3.1.2.
func (r *Reader) skipFrame() error {
	relativeOffset := 0

	if _, err := io.ReadFull(r.r, r.scratch[:4]); err != nil {
		return r.wrapNonEOFError(relativeOffset, err)
	}

	relativeOffset += 4

	size := binary.LittleEndian.Uint32(r.scratch[:4])
	if size == 0 {
		r.blockOffset += int64(relativeOffset)
		return nil
	}

	if seeker, ok := r.r.(io.Seeker); ok {
		r.blockOffset += int64(relativeOffset)
		// Implementations of Seeker do not always detect invalid offsets,
		// so check that the new offset is valid by comparing to the end.
		prev, err := seeker.Seek(0, io.SeekCurrent)
		if err != nil {
			return r.wrapError(0, err)
		}
		end, err := seeker.Seek(0, io.SeekEnd)
		if err != nil {
			return r.wrapError(0, err)
		}
		if prev > end-int64(size) {
			r.blockOffset += end - prev
			return r.makeEOFError(0)
		}

		// The new offset is valid, so seek to it.
		_, err = seeker.Seek(prev+int64(size), io.SeekStart)
		if err != nil {
			return r.wrapError(0, err)
		}
		r.blockOffset += int64(size)
		return nil
	}

	var skip []byte
	const chunk = 1 << 20 // 1M
	for size >= chunk {
		if len(skip) == 0 {
			skip = make([]byte, chunk)
		}
		if _, err := io.ReadFull(r.r, skip); err != nil {
			return r.wrapNonEOFError(relativeOffset, err)
		}
		relativeOffset += chunk
		size -= chunk
	}
	if size > 0 {
		if len(skip) == 0 {
			skip = make([]byte, size)
		}
		if _, err := io.ReadFull(r.r, skip); err != nil {
			return r.wrapNonEOFError(relativeOffset, err)
		}
		relativeOffset += int(size)
	}

	r.blockOffset += int64(relativeOffset)

	return nil
}

// readBlock reads the next block from a frame.
func (r *Reader) readBlock() error {
	relativeOffset := 0

	// Read Block_Header. RFC 3.1.1.2.
	if _, err := io.ReadFull(r.r, r.scratch[:3]); err != nil {
		return r.wrapNonEOFError(relativeOffset, err)
	}

	relativeOffset += 3

	header := uint32(r.scratch[0]) | (uint32(r.scratch[1]) << 8) | (uint32(r.scratch[2]) << 16)

	lastBlock := header&1 != 0
	blockType := (header >> 1) & 3
	blockSize := int(header >> 3)

	// Maximum block size is smaller of window size and 128K.
	// We don't record the window size for a single segment frame,
	// so just use 128K. RFC 3.1.1.2.3, 3.1.1.2.4.
	if blockSize > 128<<10 || (r.window.size > 0 && blockSize > r.window.size) {
		return r.makeError(relativeOffset, "block size too large")
	}

	// Handle different block types. RFC 3.1.1.2.2.
	switch blockType {
	case 0:
		r.setBufferSize(blockSize)
		if _, err := io.ReadFull(r.r, r.buffer); err != nil {
			return r.wrapNonEOFError(relativeOffset, err)
		}
		relativeOffset += blockSize
		r.blockOffset += int64(relativeOffset)
	case 1:
		r.setBufferSize(blockSize)
		if _, err := io.ReadFull(r.r, r.scratch[:1]); err != nil {
			return r.wrapNonEOFError(relativeOffset, err)
		}
		relativeOffset++
		v := r.scratch[0]
		for i := range r.buffer {
			r.buffer[i] = v
		}
		r.blockOffset += int64(relativeOffset)
	case 2:
		r.blockOffset += int64(relativeOffset)
		if err := r.compressedBlock(blockSize); err != nil {
			return err
		}
		r.blockOffset += int64(blockSize)
	case 3:
		return r.makeError(relativeOffset, "invalid block type")
	}

	if !r.frameSizeUnknown {
		if uint64(len(r.buffer)) > r.remainingFrameSize {
			return r.makeError(relativeOffset, "too many uncompressed bytes in frame")
		}
		r.remainingFrameSize -= uint64(len(r.buffer))
	}

	if r.hasChecksum {
		r.checksum.update(r.buffer)
	}

	if !lastBlock {
		r.window.save(r.buffer)
	} else {
		if !r.frameSizeUnknown && r.remainingFrameSize != 0 {
			return r.makeError(relativeOffset, "not enough uncompressed bytes for frame")
		}
		// Check for checksum at end of frame. RFC 3.1.1.
		if r.hasChecksum {
			if _, err := io.ReadFull(r.r, r.scratch[:4]); err != nil {
				return r.wrapNonEOFError(0, err)
			}

			inputChecksum := binary.LittleEndian.Uint32(r.scratch[:4])
			dataChecksum := uint32(r.checksum.digest())
			if inputChecksum != dataChecksum {
				return r.wrapError(0, fmt.Errorf("invalid checksum: got %#x want %#x", dataChecksum, inputChecksum))
			}

			r.blockOffset += 4
		}
		r.sawFrameHeader = false
	}

	return nil
}

// setBufferSize sets the decompressed buffer size.
// When this is called the buffer is empty.
func (r *Reader) setBufferSize(size int) {
	if cap(r.buffer) < size {
		need := size - cap(r.buffer)
		r.buffer = append(r.buffer[:cap(r.buffer)], make([]byte, need)...)
	}
	r.buffer = r.buffer[:size]
}

// zstdError is an error while decompressing.
type zstdError struct {
	offset int64
	err    error
}

func (ze *zstdError) Error() string {
	return fmt.Sprintf("zstd decompression error at %d: %v", ze.offset, ze.err)
}

func (ze *zstdError) Unwrap() error {
	return ze.err
}

func (r *Reader) makeEOFError(off int) error {
	return r.wrapError(off, io.ErrUnexpectedEOF)
}

func (r *Reader) wrapNonEOFError(off int, err error) error {
	if err == io.EOF {
		err = io.ErrUnexpectedEOF
	}
	return r.wrapError(off, err)
}

func (r *Reader) makeError(off int, msg string) error {
	return r.wrapError(off, errors.New(msg))
}

func (r *Reader) wrapError(off int, err error) error {
	if err == io.EOF {
		return err
	}
	return &zstdError{r.blockOffset + int64(off), err}
}
