// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/goarch"
	"unsafe"
)

// hexdumpWords prints a word-oriented hex dump of [p, p+len).
//
// If mark != nil, it will be passed to hexdumper.mark.
func hexdumpWords(p, len uintptr, mark func(uintptr, hexdumpMarker)) {
	printlock()

	// Provide a default annotation
	symMark := func(u uintptr, hm hexdumpMarker) {
		if mark != nil {
			mark(u, hm)
		}

		// Can we symbolize this value?
		val := *(*uintptr)(unsafe.Pointer(u))
		fn := findfunc(val)
		if fn.valid() {
			hm.start()
			print("<", funcname(fn), "+", hex(val-fn.entry()), ">\n")
		}
	}

	h := hexdumper{addr: p, mark: symMark}
	h.write(unsafe.Slice((*byte)(unsafe.Pointer(p)), len))
	h.close()
	printunlock()
}

// hexdumper is a Swiss-army knife hex dumper.
//
// To use, optionally set addr and wordBytes, then call write repeatedly,
// followed by close.
type hexdumper struct {
	// addr is the address to print for the first byte of data.
	addr uintptr

	// addrBytes is the number of bytes of addr to print. If this is 0, it
	// defaults to goarch.PtrSize.
	addrBytes uint8

	// wordBytes is the number of bytes in a word. If wordBytes is 1, this
	// prints a byte-oriented dump. If it's > 1, this interprets the data as a
	// sequence of words of the given size. If it's 0, it's treated as
	// goarch.PtrSize.
	wordBytes uint8

	// mark is an optional function that can annotate values in the hex dump.
	//
	// If non-nil, it is called with the address of every complete, aligned word
	// in the hex dump.
	//
	// If it decides to print an annotation, it must first call m.start(), then
	// print the annotation, followed by a new line.
	mark func(addr uintptr, m hexdumpMarker)

	// Below here is state

	ready int8 // 0=need to init state; 1=need to print header; 2=ready

	// dataBuf accumulates a line at a time of data, in case it's split across
	// buffers.
	dataBuf  [16]byte
	dataPos  uint8
	dataSkip uint8 // Skip first n bytes of buf on first line

	// toPos maps from byte offset in data to a visual offset in the printed line.
	toPos [16]byte
}

type hexdumpMarker struct {
	chars int
}

func (h *hexdumper) write(data []byte) {
	if h.ready == 0 {
		h.init()
	}

	// Handle leading data
	if h.dataPos > 0 {
		n := copy(h.dataBuf[h.dataPos:], data)
		h.dataPos += uint8(n)
		data = data[n:]
		if h.dataPos < uint8(len(h.dataBuf)) {
			return
		}
		h.flushLine(h.dataBuf[:])
		h.dataPos = 0
	}

	// Handle full lines in data
	for len(data) >= len(h.dataBuf) {
		h.flushLine(data[:len(h.dataBuf)])
		data = data[len(h.dataBuf):]
	}

	// Handle trailing data
	h.dataPos = uint8(copy(h.dataBuf[:], data))
}

func (h *hexdumper) close() {
	if h.dataPos > 0 {
		h.flushLine(h.dataBuf[:h.dataPos])
	}
}

func (h *hexdumper) init() {
	const bytesPerLine = len(h.dataBuf)

	if h.addrBytes == 0 {
		h.addrBytes = goarch.PtrSize
	} else if h.addrBytes < 0 || h.addrBytes > goarch.PtrSize {
		throw("invalid addrBytes")
	}

	if h.wordBytes == 0 {
		h.wordBytes = goarch.PtrSize
	}
	wb := int(h.wordBytes)
	if wb < 0 || wb >= bytesPerLine || wb&(wb-1) != 0 {
		throw("invalid wordBytes")
	}

	// Construct position mapping.
	for i := range h.toPos {
		// First, calculate the "field" within the line, applying byte swizzling.
		field := 0
		if goarch.BigEndian {
			field = i
		} else {
			field = i ^ int(wb-1)
		}
		// Translate this field into a visual offset.
		// "00112233 44556677  8899AABB CCDDEEFF"
		h.toPos[i] = byte(field*2 + field/4 + field/8)
	}

	// The first line may need to skip some fields to get to alignment.
	// Round down the starting address.
	nAddr := h.addr &^ uintptr(bytesPerLine-1)
	// Skip bytes to get to alignment.
	h.dataPos = uint8(h.addr - nAddr)
	h.dataSkip = uint8(h.addr - nAddr)
	h.addr = nAddr

	// We're ready to print the header.
	h.ready = 1
}

func (h *hexdumper) flushLine(data []byte) {
	const bytesPerLine = len(h.dataBuf)

	const maxAddrChars = 2 * goarch.PtrSize
	const addrSep = ": "
	dataStart := int(2*h.addrBytes) + len(addrSep)
	// dataChars uses the same formula to toPos above. We calculate it with the
	// "last field", then add the size of the last field.
	const dataChars = (bytesPerLine-1)*2 + (bytesPerLine-1)/4 + (bytesPerLine-1)/8 + 2
	const asciiSep = "  "
	asciiStart := dataStart + dataChars + len(asciiSep)
	const asciiChars = bytesPerLine
	nlPos := asciiStart + asciiChars

	var lineBuf [maxAddrChars + len(addrSep) + dataChars + len(asciiSep) + asciiChars + 1]byte
	clear := func() {
		for i := range lineBuf {
			lineBuf[i] = ' '
		}
	}
	clear()

	if h.ready == 1 {
		// Print column offsets header.
		for offset, pos := range h.toPos {
			h.fmtHex(lineBuf[dataStart+int(pos+1):][:1], uint64(offset))
		}
		// Print ASCII offsets.
		for offset := range asciiChars {
			h.fmtHex(lineBuf[asciiStart+offset:][:1], uint64(offset))
		}
		lineBuf[nlPos] = '\n'
		gwrite(lineBuf[:nlPos+1])
		clear()
		h.ready = 2
	}

	// Format address.
	h.fmtHex(lineBuf[:2*h.addrBytes], uint64(h.addr))
	copy(lineBuf[2*h.addrBytes:], addrSep)
	// Format data in hex and ASCII.
	for offset, b := range data {
		if offset < int(h.dataSkip) {
			continue
		}

		pos := h.toPos[offset]
		h.fmtHex(lineBuf[dataStart+int(pos):][:2], uint64(b))

		copy(lineBuf[dataStart+dataChars:], asciiSep)
		ascii := uint8('.')
		if b >= ' ' && b <= '~' {
			ascii = b
		}
		lineBuf[asciiStart+offset] = ascii
	}
	// Trim buffer.
	end := asciiStart + len(data)
	lineBuf[end] = '\n'
	buf := lineBuf[:end+1]

	// Print.
	gwrite(buf)

	// Print marks.
	if h.mark != nil {
		clear()
		for offset := 0; offset+int(h.wordBytes) <= len(data); offset += int(h.wordBytes) {
			if offset < int(h.dataSkip) {
				continue
			}
			addr := h.addr + uintptr(offset)
			// Find the position of the left edge of this word
			caret := dataStart + int(min(h.toPos[offset], h.toPos[offset+int(h.wordBytes)-1]))
			h.mark(addr, hexdumpMarker{caret})
		}
	}

	h.addr += uintptr(bytesPerLine)
	h.dataPos = 0
	h.dataSkip = 0
}

// fmtHex formats v in base 16 into buf. It fills all of buf. If buf is too
// small to represent v, it the output will start with '*'.
func (h *hexdumper) fmtHex(buf []byte, v uint64) {
	const dig = "0123456789abcdef"
	i := len(buf) - 1
	for ; i >= 0; i-- {
		buf[i] = dig[v%16]
		v /= 16
	}
	if v != 0 {
		// Indicate that we couldn't fit the whole number.
		buf[0] = '*'
	}
}

func (m hexdumpMarker) start() {
	var spaces [64]byte
	for i := range spaces {
		spaces[i] = ' '
	}
	for m.chars > len(spaces) {
		gwrite(spaces[:])
		m.chars -= len(spaces)
	}
	gwrite(spaces[:m.chars])
	print("^ ")
}
