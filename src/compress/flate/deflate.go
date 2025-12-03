// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flate

import (
	"errors"
	"fmt"
	"io"
	"math"
)

const (
	NoCompression      = 0
	BestSpeed          = 1
	BestCompression    = 9
	DefaultCompression = -1

	// HuffmanOnly disables Lempel-Ziv match searching and only performs Huffman
	// entropy encoding. This mode is useful in compressing data that has
	// already been compressed with an LZ style algorithm (e.g. Snappy or LZ4)
	// that lacks an entropy encoder. Compression gains are achieved when
	// certain bytes in the input stream occur more frequently than others.
	//
	// Note that HuffmanOnly produces a compressed output that is
	// RFC 1951 compliant. That is, any valid DEFLATE decompressor will
	// continue to be able to decompress this output.
	HuffmanOnly = -2
)

const (
	logWindowSize = 15
	windowSize    = 1 << logWindowSize
	windowMask    = windowSize - 1

	// The LZ77 step produces a sequence of literal tokens and <length, offset>
	// pair tokens. The offset is also known as distance. The underlying wire
	// format limits the range of lengths and offsets. For example, there are
	// 256 legitimate lengths: those in the range [3, 258]. This package's
	// compressor uses a higher minimum match length, enabling optimizations
	// such as finding matches via 32-bit loads and compares.
	baseMatchLength = 3       // The smallest match length per the RFC section 3.2.5
	minMatchLength  = 4       // The smallest match length that the compressor actually emits
	maxMatchLength  = 258     // The largest match length
	baseMatchOffset = 1       // The smallest match offset
	maxMatchOffset  = 1 << 15 // The largest match offset

	// The maximum number of tokens we put into a single flate block, just to
	// stop things from getting too large.
	maxFlateBlockTokens = 1 << 14
	maxStoreBlockSize   = 65535
	hashBits            = 17 // After 17 performance degrades
	hashSize            = 1 << hashBits
	hashMask            = (1 << hashBits) - 1
	maxHashOffset       = 1 << 24

	skipNever = math.MaxInt32
)

type compressionLevel struct {
	level, good, lazy, nice, chain, fastSkipHashing int
}

var levels = []compressionLevel{
	{0, 0, 0, 0, 0, 0}, // NoCompression.
	{1, 0, 0, 0, 0, 0}, // BestSpeed uses a custom algorithm; see deflatefast.go.
	// For levels 2-3 we don't bother trying with lazy matches.
	{2, 4, 0, 16, 8, 5},
	{3, 4, 0, 32, 32, 6},
	// Levels 4-9 use increasingly more lazy matching
	// and increasingly stringent conditions for "good enough".
	{4, 4, 4, 16, 16, skipNever},
	{5, 8, 16, 32, 32, skipNever},
	{6, 8, 16, 128, 128, skipNever},
	{7, 8, 32, 128, 256, skipNever},
	{8, 32, 128, 258, 1024, skipNever},
	{9, 32, 258, 258, 4096, skipNever},
}

type compressor struct {
	compressionLevel

	w          *huffmanBitWriter
	bulkHasher func([]byte, []uint32)

	// compression algorithm
	fill      func(*compressor, []byte) int // copy data to window
	step      func(*compressor)             // process window
	bestSpeed *deflateFast                  // Encoder for BestSpeed

	// input window: unprocessed data is window[index:windowEnd]
	index         int
	window        []byte
	windowEnd     int
	blockStart    int  // window index where current tokens start
	byteAvailable bool // if true, still need to process window[index-1].

	sync bool // requesting flush

	// queued output tokens
	tokens []token

	// deflate state
	length         int
	offset         int
	maxInsertIndex int
	err            error

	// Input hash chains
	// hashHead[hashValue] contains the largest inputIndex with the specified hash value
	// If hashHead[hashValue] is within the current window, then
	// hashPrev[hashHead[hashValue] & windowMask] contains the previous index
	// with the same hash value.
	// These are large and do not contain pointers, so put them
	// near the end of the struct so the GC has to scan less.
	chainHead  int
	hashHead   [hashSize]uint32
	hashPrev   [windowSize]uint32
	hashOffset int

	// hashMatch must be able to contain hashes for the maximum match length.
	hashMatch [maxMatchLength - 1]uint32
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
		d.hashOffset += windowSize
		if d.hashOffset > maxHashOffset {
			delta := d.hashOffset - 1
			d.hashOffset -= delta
			d.chainHead -= delta

			// Iterate over slices instead of arrays to avoid copying
			// the entire table onto the stack (Issue #18625).
			for i, v := range d.hashPrev[:] {
				if int(v) > delta {
					d.hashPrev[i] = uint32(int(v) - delta)
				} else {
					d.hashPrev[i] = 0
				}
			}
			for i, v := range d.hashHead[:] {
				if int(v) > delta {
					d.hashHead[i] = uint32(int(v) - delta)
				} else {
					d.hashHead[i] = 0
				}
			}
		}
	}
	n := copy(d.window[d.windowEnd:], b)
	d.windowEnd += n
	return n
}

func (d *compressor) writeBlock(tokens []token, index int) error {
	if index > 0 {
		var window []byte
		if d.blockStart <= index {
			window = d.window[d.blockStart:index]
		}
		d.blockStart = index
		d.w.writeBlock(tokens, false, window)
		return d.w.err
	}
	return nil
}

// fillWindow will fill the current window with the supplied
// dictionary and calculate all hashes.
// This is much faster than doing a full encode.
// Should only be used after a reset.
func (d *compressor) fillWindow(b []byte) {
	// Do not fill window if we are in store-only mode.
	if d.compressionLevel.level < 2 {
		return
	}
	if d.index != 0 || d.windowEnd != 0 {
		panic("internal error: fillWindow called with stale data")
	}

	// If we are given too much, cut it.
	if len(b) > windowSize {
		b = b[len(b)-windowSize:]
	}
	// Add all to window.
	n := copy(d.window, b)

	// Calculate 256 hashes at the time (more L1 cache hits)
	loops := (n + 256 - minMatchLength) / 256
	for j := 0; j < loops; j++ {
		index := j * 256
		end := index + 256 + minMatchLength - 1
		if end > n {
			end = n
		}
		toCheck := d.window[index:end]
		dstSize := len(toCheck) - minMatchLength + 1

		if dstSize <= 0 {
			continue
		}

		dst := d.hashMatch[:dstSize]
		d.bulkHasher(toCheck, dst)
		for i, val := range dst {
			di := i + index
			hh := &d.hashHead[val&hashMask]
			// Get previous value with the same hash.
			// Our chain should point to the previous value.
			d.hashPrev[di&windowMask] = *hh
			// Set the head of the hash chain to us.
			*hh = uint32(di + d.hashOffset)
		}
	}
	// Update window information.
	d.windowEnd = n
	d.index = n
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

	wEnd := win[pos+length]
	wPos := win[pos:]
	minIndex := pos - windowSize

	for i := prevHead; tries > 0; tries-- {
		if wEnd == win[i+length] {
			n := matchLen(win[i:], wPos, minMatchLook)

			if n > length && (n > minMatchLength || pos-i <= 4096) {
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
		i = int(d.hashPrev[i&windowMask]) - d.hashOffset
		if i < minIndex || i < 0 {
			break
		}
	}
	return
}

func (d *compressor) writeStoredBlock(buf []byte) error {
	if d.w.writeStoredHeader(len(buf), false); d.w.err != nil {
		return d.w.err
	}
	d.w.writeBytes(buf)
	return d.w.err
}

const hashmul = 0x1e35a7bd

// hash4 returns a hash representation of the first 4 bytes
// of the supplied slice.
// The caller must ensure that len(b) >= 4.
func hash4(b []byte) uint32 {
	return ((uint32(b[3]) | uint32(b[2])<<8 | uint32(b[1])<<16 | uint32(b[0])<<24) * hashmul) >> (32 - hashBits)
}

// bulkHash4 will compute hashes using the same
// algorithm as hash4.
func bulkHash4(b []byte, dst []uint32) {
	if len(b) < minMatchLength {
		return
	}
	hb := uint32(b[3]) | uint32(b[2])<<8 | uint32(b[1])<<16 | uint32(b[0])<<24
	dst[0] = (hb * hashmul) >> (32 - hashBits)
	end := len(b) - minMatchLength + 1
	for i := 1; i < end; i++ {
		hb = (hb << 8) | uint32(b[i+3])
		dst[i] = (hb * hashmul) >> (32 - hashBits)
	}
}

// matchLen returns the number of matching bytes in a and b
// up to length 'max'. Both slices must be at least 'max'
// bytes in size.
func matchLen(a, b []byte, max int) int {
	a = a[:max]
	b = b[:len(a)]
	for i, av := range a {
		if b[i] != av {
			return i
		}
	}
	return max
}

// encSpeed will compress and store the currently added data,
// if enough has been accumulated or we at the end of the stream.
// Any error that occurred will be in d.err
func (d *compressor) encSpeed() {
	// We only compress if we have maxStoreBlockSize.
	if d.windowEnd < maxStoreBlockSize {
		if !d.sync {
			return
		}

		// Handle small sizes.
		if d.windowEnd < 128 {
			switch {
			case d.windowEnd == 0:
				return
			case d.windowEnd <= 16:
				d.err = d.writeStoredBlock(d.window[:d.windowEnd])
			default:
				d.w.writeBlockHuff(false, d.window[:d.windowEnd])
				d.err = d.w.err
			}
			d.windowEnd = 0
			d.bestSpeed.reset()
			return
		}

	}
	// Encode the block.
	d.tokens = d.bestSpeed.encode(d.tokens[:0], d.window[:d.windowEnd])

	// If we removed less than 1/16th, Huffman compress the block.
	if len(d.tokens) > d.windowEnd-(d.windowEnd>>4) {
		d.w.writeBlockHuff(false, d.window[:d.windowEnd])
	} else {
		d.w.writeBlockDynamic(d.tokens, false, d.window[:d.windowEnd])
	}
	d.err = d.w.err
	d.windowEnd = 0
}

func (d *compressor) initDeflate() {
	d.window = make([]byte, 2*windowSize)
	d.hashOffset = 1
	d.tokens = make([]token, 0, maxFlateBlockTokens+1)
	d.length = minMatchLength - 1
	d.offset = 0
	d.byteAvailable = false
	d.index = 0
	d.chainHead = -1
	d.bulkHasher = bulkHash4
}

func (d *compressor) deflate() {
	if d.windowEnd-d.index < minMatchLength+maxMatchLength && !d.sync {
		return
	}

	d.maxInsertIndex = d.windowEnd - (minMatchLength - 1)

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
					d.tokens = append(d.tokens, literalToken(uint32(d.window[d.index-1])))
					d.byteAvailable = false
				}
				if len(d.tokens) > 0 {
					if d.err = d.writeBlock(d.tokens, d.index); d.err != nil {
						return
					}
					d.tokens = d.tokens[:0]
				}
				break Loop
			}
		}
		if d.index < d.maxInsertIndex {
			// Update the hash
			hash := hash4(d.window[d.index : d.index+minMatchLength])
			hh := &d.hashHead[hash&hashMask]
			d.chainHead = int(*hh)
			d.hashPrev[d.index&windowMask] = uint32(d.chainHead)
			*hh = uint32(d.index + d.hashOffset)
		}
		prevLength := d.length
		prevOffset := d.offset
		d.length = minMatchLength - 1
		d.offset = 0
		minIndex := d.index - windowSize
		if minIndex < 0 {
			minIndex = 0
		}

		if d.chainHead-d.hashOffset >= minIndex &&
			(d.fastSkipHashing != skipNever && lookahead > minMatchLength-1 ||
				d.fastSkipHashing == skipNever && lookahead > prevLength && prevLength < d.lazy) {
			if newLength, newOffset, ok := d.findMatch(d.index, d.chainHead-d.hashOffset, minMatchLength-1, lookahead); ok {
				d.length = newLength
				d.offset = newOffset
			}
		}
		if d.fastSkipHashing != skipNever && d.length >= minMatchLength ||
			d.fastSkipHashing == skipNever && prevLength >= minMatchLength && d.length <= prevLength {
			// There was a match at the previous step, and the current match is
			// not better. Output the previous match.
			if d.fastSkipHashing != skipNever {
				d.tokens = append(d.tokens, matchToken(uint32(d.length-baseMatchLength), uint32(d.offset-baseMatchOffset)))
			} else {
				d.tokens = append(d.tokens, matchToken(uint32(prevLength-baseMatchLength), uint32(prevOffset-baseMatchOffset)))
			}
			// Insert in the hash table all strings up to the end of the match.
			// index and index-1 are already inserted. If there is not enough
			// lookahead, the last two strings are not inserted into the hash
			// table.
			if d.length <= d.fastSkipHashing {
				var newIndex int
				if d.fastSkipHashing != skipNever {
					newIndex = d.index + d.length
				} else {
					newIndex = d.index + prevLength - 1
				}
				index := d.index
				for index++; index < newIndex; index++ {
					if index < d.maxInsertIndex {
						hash := hash4(d.window[index : index+minMatchLength])
						// Get previous value with the same hash.
						// Our chain should point to the previous value.
						hh := &d.hashHead[hash&hashMask]
						d.hashPrev[index&windowMask] = *hh
						// Set the head of the hash chain to us.
						*hh = uint32(index + d.hashOffset)
					}
				}
				d.index = index

				if d.fastSkipHashing == skipNever {
					d.byteAvailable = false
					d.length = minMatchLength - 1
				}
			} else {
				// For matches this long, we don't bother inserting each individual
				// item into the table.
				d.index += d.length
			}
			if len(d.tokens) == maxFlateBlockTokens {
				// The block includes the current character
				if d.err = d.writeBlock(d.tokens, d.index); d.err != nil {
					return
				}
				d.tokens = d.tokens[:0]
			}
		} else {
			if d.fastSkipHashing != skipNever || d.byteAvailable {
				i := d.index - 1
				if d.fastSkipHashing != skipNever {
					i = d.index
				}
				d.tokens = append(d.tokens, literalToken(uint32(d.window[i])))
				if len(d.tokens) == maxFlateBlockTokens {
					if d.err = d.writeBlock(d.tokens, i+1); d.err != nil {
						return
					}
					d.tokens = d.tokens[:0]
				}
			}
			d.index++
			if d.fastSkipHashing == skipNever {
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
	if d.windowEnd > 0 && (d.windowEnd == maxStoreBlockSize || d.sync) {
		d.err = d.writeStoredBlock(d.window[:d.windowEnd])
		d.windowEnd = 0
	}
}

// storeHuff compresses and stores the currently added data
// when the d.window is full or we are at the end of the stream.
// Any error that occurred will be in d.err
func (d *compressor) storeHuff() {
	if d.windowEnd < len(d.window) && !d.sync || d.windowEnd == 0 {
		return
	}
	d.w.writeBlockHuff(false, d.window[:d.windowEnd])
	d.err = d.w.err
	d.windowEnd = 0
}

func (d *compressor) write(b []byte) (n int, err error) {
	if d.err != nil {
		return 0, d.err
	}
	n = len(b)
	for len(b) > 0 {
		d.step(d)
		b = b[d.fill(d, b):]
		if d.err != nil {
			return 0, d.err
		}
	}
	return n, nil
}

func (d *compressor) syncFlush() error {
	if d.err != nil {
		return d.err
	}
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

func (d *compressor) init(w io.Writer, level int) (err error) {
	d.w = newHuffmanBitWriter(w)

	switch {
	case level == NoCompression:
		d.window = make([]byte, maxStoreBlockSize)
		d.fill = (*compressor).fillStore
		d.step = (*compressor).store
	case level == HuffmanOnly:
		d.window = make([]byte, maxStoreBlockSize)
		d.fill = (*compressor).fillStore
		d.step = (*compressor).storeHuff
	case level == BestSpeed:
		d.compressionLevel = levels[level]
		d.window = make([]byte, maxStoreBlockSize)
		d.fill = (*compressor).fillStore
		d.step = (*compressor).encSpeed
		d.bestSpeed = newDeflateFast()
		d.tokens = make([]token, maxStoreBlockSize)
	case level == DefaultCompression:
		level = 6
		fallthrough
	case 2 <= level && level <= 9:
		d.compressionLevel = levels[level]
		d.initDeflate()
		d.fill = (*compressor).fillDeflate
		d.step = (*compressor).deflate
	default:
		return fmt.Errorf("flate: invalid compression level %d: want value in range [-2, 9]", level)
	}
	return nil
}

func (d *compressor) reset(w io.Writer) {
	d.w.reset(w)
	d.sync = false
	d.err = nil
	switch d.compressionLevel.level {
	case NoCompression:
		d.windowEnd = 0
	case BestSpeed:
		d.windowEnd = 0
		d.tokens = d.tokens[:0]
		d.bestSpeed.reset()
	default:
		d.chainHead = -1
		clear(d.hashHead[:])
		clear(d.hashPrev[:])
		d.hashOffset = 1
		d.index, d.windowEnd = 0, 0
		d.blockStart, d.byteAvailable = 0, false
		d.tokens = d.tokens[:0]
		d.length = minMatchLength - 1
		d.offset = 0
		d.maxInsertIndex = 0
	}
}

func (d *compressor) close() error {
	if d.err == errWriterClosed {
		return nil
	}
	if d.err != nil {
		return d.err
	}
	d.sync = true
	d.step(d)
	if d.err != nil {
		return d.err
	}
	if d.w.writeStoredHeader(0, true); d.w.err != nil {
		return d.w.err
	}
	d.w.flush()
	if d.w.err != nil {
		return d.w.err
	}
	d.err = errWriterClosed
	return nil
}

// NewWriter returns a new [Writer] compressing data at the given level.
// Following zlib, levels range from 1 ([BestSpeed]) to 9 ([BestCompression]);
// higher levels typically run slower but compress more. Level 0
// ([NoCompression]) does not attempt any compression; it only adds the
// necessary DEFLATE framing.
// Level -1 ([DefaultCompression]) uses the default compression level.
// Level -2 ([HuffmanOnly]) will use Huffman compression only, giving
// a very fast compression for all types of input, but sacrificing considerable
// compression efficiency.
//
// If level is in the range [-2, 9] then the error returned will be nil.
// Otherwise the error returned will be non-nil.
func NewWriter(w io.Writer, level int) (*Writer, error) {
	var dw Writer
	if err := dw.d.init(w, level); err != nil {
		return nil, err
	}
	return &dw, nil
}

// NewWriterDict is like [NewWriter] but initializes the new
// [Writer] with a preset dictionary. The returned [Writer] behaves
// as if the dictionary had been written to it without producing
// any compressed output. The compressed data written to w
// can only be decompressed by a reader initialized with the
// same dictionary (see [NewReaderDict]).
func NewWriterDict(w io.Writer, level int, dict []byte) (*Writer, error) {
	dw := &dictWriter{w}
	zw, err := NewWriter(dw, level)
	if err != nil {
		return nil, err
	}
	zw.d.fillWindow(dict)
	zw.dict = append(zw.dict, dict...) // duplicate dictionary for Reset method.
	return zw, nil
}

type dictWriter struct {
	w io.Writer
}

func (w *dictWriter) Write(b []byte) (n int, err error) {
	return w.w.Write(b)
}

var errWriterClosed = errors.New("flate: closed writer")

// A Writer takes data written to it and writes the compressed
// form of that data to an underlying writer (see [NewWriter]).
type Writer struct {
	d    compressor
	dict []byte
}

// Write writes data to w, which will eventually write the
// compressed form of data to its underlying writer.
func (w *Writer) Write(data []byte) (n int, err error) {
	return w.d.write(data)
}

// Flush flushes any pending data to the underlying writer.
// It is useful mainly in compressed network protocols, to ensure that
// a remote reader has enough data to reconstruct a packet.
// Flush does not return until the data has been written.
// Calling Flush when there is no pending data still causes the [Writer]
// to emit a sync marker of at least 4 bytes.
// If the underlying writer returns an error, Flush returns that error.
//
// In the terminology of the zlib library, Flush is equivalent to Z_SYNC_FLUSH.
func (w *Writer) Flush() error {
	// For more about flushing:
	// https://www.bolet.org/~pornin/deflate-flush.html
	return w.d.syncFlush()
}

// Close flushes and closes the writer.
func (w *Writer) Close() error {
	return w.d.close()
}

// Reset discards the writer's state and makes it equivalent to
// the result of [NewWriter] or [NewWriterDict] called with dst
// and w's level and dictionary.
func (w *Writer) Reset(dst io.Writer) {
	if dw, ok := w.d.w.writer.(*dictWriter); ok {
		// w was created with NewWriterDict
		dw.w = dst
		w.d.reset(dw)
		w.d.fillWindow(w.dict)
	} else {
		// w was created with NewWriter
		w.d.reset(dst)
	}
}
