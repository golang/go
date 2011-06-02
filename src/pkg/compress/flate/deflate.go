// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flate

import (
	"io"
	"math"
	"os"
)

const (
	NoCompression      = 0
	BestSpeed          = 1
	fastCompression    = 3
	BestCompression    = 9
	DefaultCompression = -1
	logWindowSize      = 15
	windowSize         = 1 << logWindowSize
	windowMask         = windowSize - 1
	logMaxOffsetSize   = 15  // Standard DEFLATE
	minMatchLength     = 3   // The smallest match that the compressor looks for
	maxMatchLength     = 258 // The longest match for the compressor
	minOffsetSize      = 1   // The shortest offset that makes any sence

	// The maximum number of tokens we put into a single flat block, just too
	// stop things from getting too large.
	maxFlateBlockTokens = 1 << 14
	maxStoreBlockSize   = 65535
	hashBits            = 15
	hashSize            = 1 << hashBits
	hashMask            = (1 << hashBits) - 1
	hashShift           = (hashBits + minMatchLength - 1) / minMatchLength
)

type compressionLevel struct {
	good, lazy, nice, chain, fastSkipHashing int
}

var levels = []compressionLevel{
	{}, // 0
	// For levels 1-3 we don't bother trying with lazy matches
	{3, 0, 8, 4, 4},
	{3, 0, 16, 8, 5},
	{3, 0, 32, 32, 6},
	// Levels 4-9 use increasingly more lazy matching
	// and increasingly stringent conditions for "good enough".
	{4, 4, 16, 16, math.MaxInt32},
	{8, 16, 32, 32, math.MaxInt32},
	{8, 16, 128, 128, math.MaxInt32},
	{8, 32, 128, 256, math.MaxInt32},
	{32, 128, 258, 1024, math.MaxInt32},
	{32, 258, 258, 4096, math.MaxInt32},
}

type compressor struct {
	compressionLevel

	w *huffmanBitWriter

	// compression algorithm
	fill func(*compressor, []byte) int // copy data to window
	step func(*compressor)             // process window
	sync bool                          // requesting flush

	// Input hash chains
	// hashHead[hashValue] contains the largest inputIndex with the specified hash value
	// If hashHead[hashValue] is within the current window, then
	// hashPrev[hashHead[hashValue] & windowMask] contains the previous index
	// with the same hash value.
	chainHead int
	hashHead  []int
	hashPrev  []int

	// input window: unprocessed data is window[index:windowEnd]
	index         int
	window        []byte
	windowEnd     int
	blockStart    int  // window index where current tokens start
	byteAvailable bool // if true, still need to process window[index-1].

	// queued output tokens: tokens[:ti]
	tokens []token
	ti     int

	// deflate state
	length         int
	offset         int
	hash           int
	maxInsertIndex int
	err            os.Error
}

func (d *compressor) fillDeflate(b []byte) int {
	if d.index >= 2*windowSize-(minMatchLength+maxMatchLength) {
		// shift the window by windowSize
		copy(d.window, d.window[windowSize:2*windowSize])
		d.index -= windowSize
		d.windowEnd -= windowSize
		if d.blockStart >= windowSize {
			d.blockStart -= windowSize
		} else {
			d.blockStart = math.MaxInt32
		}
		for i, h := range d.hashHead {
			v := h - windowSize
			if v < -1 {
				v = -1
			}
			d.hashHead[i] = v
		}
		for i, h := range d.hashPrev {
			v := -h - windowSize
			if v < -1 {
				v = -1
			}
			d.hashPrev[i] = v
		}
	}
	n := copy(d.window[d.windowEnd:], b)
	d.windowEnd += n
	return n
}

func (d *compressor) writeBlock(tokens []token, index int, eof bool) os.Error {
	if index > 0 || eof {
		var window []byte
		if d.blockStart <= index {
			window = d.window[d.blockStart:index]
		}
		d.blockStart = index
		d.w.writeBlock(tokens, eof, window)
		return d.w.err
	}
	return nil
}

// Try to find a match starting at index whose length is greater than prevSize.
// We only look at chainCount possibilities before giving up.
func (d *compressor) findMatch(pos int, prevHead int, prevLength int, lookahead int) (length, offset int, ok bool) {
	minMatchLook := maxMatchLength
	if lookahead < minMatchLook {
		minMatchLook = lookahead
	}

	win := d.window[0 : pos+minMatchLook]

	// We quit when we get a match that's at least nice long
	nice := len(win) - pos
	if d.nice < nice {
		nice = d.nice
	}

	// If we've got a match that's good enough, only look in 1/4 the chain.
	tries := d.chain
	length = prevLength
	if length >= d.good {
		tries >>= 2
	}

	w0 := win[pos]
	w1 := win[pos+1]
	wEnd := win[pos+length]
	minIndex := pos - windowSize

	for i := prevHead; tries > 0; tries-- {
		if w0 == win[i] && w1 == win[i+1] && wEnd == win[i+length] {
			// The hash function ensures that if win[i] and win[i+1] match, win[i+2] matches

			n := 3
			for pos+n < len(win) && win[i+n] == win[pos+n] {
				n++
			}
			if n > length && (n > 3 || pos-i <= 4096) {
				length = n
				offset = pos - i
				ok = true
				if n >= nice {
					// The match is good enough that we don't try to find a better one.
					break
				}
				wEnd = win[pos+n]
			}
		}
		if i == minIndex {
			// hashPrev[i & windowMask] has already been overwritten, so stop now.
			break
		}
		if i = d.hashPrev[i&windowMask]; i < minIndex || i < 0 {
			break
		}
	}
	return
}

func (d *compressor) writeStoredBlock(buf []byte) os.Error {
	if d.w.writeStoredHeader(len(buf), false); d.w.err != nil {
		return d.w.err
	}
	d.w.writeBytes(buf)
	return d.w.err
}

func (d *compressor) initDeflate() {
	d.hashHead = make([]int, hashSize)
	d.hashPrev = make([]int, windowSize)
	d.window = make([]byte, 2*windowSize)
	fillInts(d.hashHead, -1)
	d.tokens = make([]token, maxFlateBlockTokens, maxFlateBlockTokens+1)
	d.length = minMatchLength - 1
	d.offset = 0
	d.byteAvailable = false
	d.index = 0
	d.ti = 0
	d.hash = 0
	d.chainHead = -1
}

func (d *compressor) deflate() {
	if d.windowEnd-d.index < minMatchLength+maxMatchLength && !d.sync {
		return
	}

	d.maxInsertIndex = d.windowEnd - (minMatchLength - 1)
	if d.index < d.maxInsertIndex {
		d.hash = int(d.window[d.index])<<hashShift + int(d.window[d.index+1])
	}

Loop:
	for {
		if d.index > d.windowEnd {
			panic("index > windowEnd")
		}
		lookahead := d.windowEnd - d.index
		if lookahead < minMatchLength+maxMatchLength {
			if !d.sync {
				break Loop
			}
			if d.index > d.windowEnd {
				panic("index > windowEnd")
			}
			if lookahead == 0 {
				// Flush current output block if any.
				if d.byteAvailable {
					// There is still one pending token that needs to be flushed
					d.tokens[d.ti] = literalToken(uint32(d.window[d.index-1]))
					d.ti++
					d.byteAvailable = false
				}
				if d.ti > 0 {
					if d.err = d.writeBlock(d.tokens[0:d.ti], d.index, false); d.err != nil {
						return
					}
					d.ti = 0
				}
				break Loop
			}
		}
		if d.index < d.maxInsertIndex {
			// Update the hash
			d.hash = (d.hash<<hashShift + int(d.window[d.index+2])) & hashMask
			d.chainHead = d.hashHead[d.hash]
			d.hashPrev[d.index&windowMask] = d.chainHead
			d.hashHead[d.hash] = d.index
		}
		prevLength := d.length
		prevOffset := d.offset
		d.length = minMatchLength - 1
		d.offset = 0
		minIndex := d.index - windowSize
		if minIndex < 0 {
			minIndex = 0
		}

		if d.chainHead >= minIndex &&
			(d.fastSkipHashing != 0 && lookahead > minMatchLength-1 ||
				d.fastSkipHashing == 0 && lookahead > prevLength && prevLength < d.lazy) {
			if newLength, newOffset, ok := d.findMatch(d.index, d.chainHead, minMatchLength-1, lookahead); ok {
				d.length = newLength
				d.offset = newOffset
			}
		}
		if d.fastSkipHashing != 0 && d.length >= minMatchLength ||
			d.fastSkipHashing == 0 && prevLength >= minMatchLength && d.length <= prevLength {
			// There was a match at the previous step, and the current match is
			// not better. Output the previous match.
			if d.fastSkipHashing != 0 {
				d.tokens[d.ti] = matchToken(uint32(d.length-minMatchLength), uint32(d.offset-minOffsetSize))
			} else {
				d.tokens[d.ti] = matchToken(uint32(prevLength-minMatchLength), uint32(prevOffset-minOffsetSize))
			}
			d.ti++
			// Insert in the hash table all strings up to the end of the match.
			// index and index-1 are already inserted. If there is not enough
			// lookahead, the last two strings are not inserted into the hash
			// table.
			if d.length <= d.fastSkipHashing {
				var newIndex int
				if d.fastSkipHashing != 0 {
					newIndex = d.index + d.length
				} else {
					newIndex = prevLength - 1
				}
				for d.index++; d.index < newIndex; d.index++ {
					if d.index < d.maxInsertIndex {
						d.hash = (d.hash<<hashShift + int(d.window[d.index+2])) & hashMask
						// Get previous value with the same hash.
						// Our chain should point to the previous value.
						d.hashPrev[d.index&windowMask] = d.hashHead[d.hash]
						// Set the head of the hash chain to us.
						d.hashHead[d.hash] = d.index
					}
				}
				if d.fastSkipHashing == 0 {
					d.byteAvailable = false
					d.length = minMatchLength - 1
				}
			} else {
				// For matches this long, we don't bother inserting each individual
				// item into the table.
				d.index += d.length
				d.hash = (int(d.window[d.index])<<hashShift + int(d.window[d.index+1]))
			}
			if d.ti == maxFlateBlockTokens {
				// The block includes the current character
				if d.err = d.writeBlock(d.tokens, d.index, false); d.err != nil {
					return
				}
				d.ti = 0
			}
		} else {
			if d.fastSkipHashing != 0 || d.byteAvailable {
				i := d.index - 1
				if d.fastSkipHashing != 0 {
					i = d.index
				}
				d.tokens[d.ti] = literalToken(uint32(d.window[i]))
				d.ti++
				if d.ti == maxFlateBlockTokens {
					if d.err = d.writeBlock(d.tokens, i+1, false); d.err != nil {
						return
					}
					d.ti = 0
				}
			}
			d.index++
			if d.fastSkipHashing == 0 {
				d.byteAvailable = true
			}
		}
	}
}

func (d *compressor) fillStore(b []byte) int {
	n := copy(d.window[d.windowEnd:], b)
	d.windowEnd += n
	return n
}

func (d *compressor) store() {
	if d.windowEnd > 0 {
		d.err = d.writeStoredBlock(d.window[:d.windowEnd])
	}
	d.windowEnd = 0
}

func (d *compressor) write(b []byte) (n int, err os.Error) {
	n = len(b)
	b = b[d.fill(d, b):]
	for len(b) > 0 {
		d.step(d)
		b = b[d.fill(d, b):]
	}
	return n, d.err
}

func (d *compressor) syncFlush() os.Error {
	d.sync = true
	d.step(d)
	if d.err == nil {
		d.w.writeStoredHeader(0, false)
		d.w.flush()
		d.err = d.w.err
	}
	d.sync = false
	return d.err
}

func (d *compressor) init(w io.Writer, level int) (err os.Error) {
	d.w = newHuffmanBitWriter(w)

	switch {
	case level == NoCompression:
		d.window = make([]byte, maxStoreBlockSize)
		d.fill = (*compressor).fillStore
		d.step = (*compressor).store
	case level == DefaultCompression:
		level = 6
		fallthrough
	case 1 <= level && level <= 9:
		d.compressionLevel = levels[level]
		d.initDeflate()
		d.fill = (*compressor).fillDeflate
		d.step = (*compressor).deflate
	default:
		return WrongValueError{"level", 0, 9, int32(level)}
	}
	return nil
}

func (d *compressor) close() os.Error {
	d.sync = true
	d.step(d)
	if d.err != nil {
		return d.err
	}
	if d.w.writeStoredHeader(0, true); d.w.err != nil {
		return d.w.err
	}
	d.w.flush()
	return d.w.err
}

// NewWriter returns a new Writer compressing
// data at the given level.  Following zlib, levels
// range from 1 (BestSpeed) to 9 (BestCompression);
// higher levels typically run slower but compress more.
// Level 0 (NoCompression) does not attempt any
// compression; it only adds the necessary DEFLATE framing.
func NewWriter(w io.Writer, level int) *Writer {
	const logWindowSize = logMaxOffsetSize
	var dw Writer
	dw.d.init(w, level)
	return &dw
}

// NewWriterDict is like NewWriter but initializes the new
// Writer with a preset dictionary.  The returned Writer behaves
// as if the dictionary had been written to it without producing
// any compressed output.  The compressed data written to w
// can only be decompressed by a Reader initialized with the
// same dictionary.
func NewWriterDict(w io.Writer, level int, dict []byte) *Writer {
	dw := &dictWriter{w, false}
	zw := NewWriter(dw, level)
	zw.Write(dict)
	zw.Flush()
	dw.enabled = true
	return zw
}

type dictWriter struct {
	w       io.Writer
	enabled bool
}

func (w *dictWriter) Write(b []byte) (n int, err os.Error) {
	if w.enabled {
		return w.w.Write(b)
	}
	return len(b), nil
}

// A Writer takes data written to it and writes the compressed
// form of that data to an underlying writer (see NewWriter).
type Writer struct {
	d compressor
}

// Write writes data to w, which will eventually write the
// compressed form of data to its underlying writer.
func (w *Writer) Write(data []byte) (n int, err os.Error) {
	return w.d.write(data)
}

// Flush flushes any pending compressed data to the underlying writer.
// It is useful mainly in compressed network protocols, to ensure that
// a remote reader has enough data to reconstruct a packet.
// Flush does not return until the data has been written.
// If the underlying writer returns an error, Flush returns that error.
//
// In the terminology of the zlib library, Flush is equivalent to Z_SYNC_FLUSH.
func (w *Writer) Flush() os.Error {
	// For more about flushing:
	// http://www.bolet.org/~pornin/deflate-flush.html
	return w.d.syncFlush()
}

// Close flushes and closes the writer.
func (w *Writer) Close() os.Error {
	return w.d.close()
}
