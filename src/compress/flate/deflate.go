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

	logWindowSize  = 15
	windowSize     = 1 << logWindowSize
	windowMask     = windowSize - 1
	minMatchLength = 4   // The smallest match that the compressor looks for
	maxMatchLength = 258 // The longest match for the compressor
	minOffsetSize  = 1   // The shortest offset that makes any sense

	// The maximum number of tokens we will encode at the time.
	// Smaller sizes usually creates less optimal blocks.
	// Bigger can make context switching slow.
	// We use this for levels 7-9, so we make it big.
	maxFlateBlockTokens = 1 << 15
	maxStoreBlockSize   = 65535
	hashBits            = 17 // After 17 performance degrades
	hashSize            = 1 << hashBits
	hashMask            = (1 << hashBits) - 1
	maxHashOffset       = 1 << 28

	skipNever = math.MaxInt32
)

type compressionLevel struct {
	good, lazy, nice, chain, level int
}

var levels = []compressionLevel{
	{}, // 0
	// Level 1-6 uses specialized algorithm - values not used
	{0, 0, 0, 0, 1},
	{0, 0, 0, 0, 2},
	{0, 0, 0, 0, 3},
	{0, 0, 0, 0, 4},
	{0, 0, 0, 0, 5},
	{0, 0, 0, 0, 6},
	// Levels 7-9 use increasingly more lazy matching
	// and increasingly stringent conditions for "good enough".
	{8, 12, 16, 24, 7},
	{16, 30, 40, 64, 8},
	{32, 258, 258, 1024, 9},
}

// advancedState contains state for the advanced levels, with bigger hash tables, etc.
type advancedState struct {
	// deflate state
	length         int
	offset         int
	maxInsertIndex int
	chainHead      int
	hashOffset     int

	ii uint16 // position of last match, intended to overflow to reset.

	// input window: unprocessed data is window[index:windowEnd]
	index     int
	hashMatch [maxMatchLength + minMatchLength]uint32

	// Input hash chains
	// hashHead[hashValue] contains the largest inputIndex with the specified hash value
	// If hashHead[hashValue] is within the current window, then
	// hashPrev[hashHead[hashValue] & windowMask] contains the previous index
	// with the same hash value.
	hashHead [hashSize]uint32
	hashPrev [windowSize]uint32
}

type compressor struct {
	compressionLevel

	h *huffmanEncoder
	w *huffmanBitWriter

	// compression algorithm
	fill func(*compressor, []byte) int // copy data to window
	step func(*compressor)             // process window

	window     []byte
	windowEnd  int
	blockStart int // window index where current tokens start
	err        error

	// queued output tokens
	tokens tokens
	fast   fastEnc
	state  *advancedState

	sync          bool // requesting flush
	byteAvailable bool // if true, still need to process window[index-1].
}

func (d *compressor) fillDeflate(b []byte) int {
	s := d.state
	if s.index >= 2*windowSize-(minMatchLength+maxMatchLength) {
		// shift the window by windowSize
		//copy(d.window[:], d.window[windowSize:2*windowSize])
		*(*[windowSize]byte)(d.window) = *(*[windowSize]byte)(d.window[windowSize:])
		s.index -= windowSize
		d.windowEnd -= windowSize
		if d.blockStart >= windowSize {
			d.blockStart -= windowSize
		} else {
			d.blockStart = math.MaxInt32
		}
		s.hashOffset += windowSize
		if s.hashOffset > maxHashOffset {
			delta := s.hashOffset - 1
			s.hashOffset -= delta
			s.chainHead -= delta
			// Iterate over slices instead of arrays to avoid copying
			// the entire table onto the stack (Issue #18625).
			for i, v := range s.hashPrev[:] {
				s.hashPrev[i] = uint32(max(int(v)-delta, 0))
			}
			for i, v := range s.hashHead[:] {
				s.hashHead[i] = uint32(max(int(v)-delta, 0))
			}
		}
	}
	n := copy(d.window[d.windowEnd:], b)
	d.windowEnd += n
	return n
}

func (d *compressor) writeBlock(tok *tokens, index int, eof bool) error {
	if index > 0 || eof {
		var window []byte
		if d.blockStart <= index {
			window = d.window[d.blockStart:index]
		}
		d.blockStart = index
		d.w.writeBlockDynamic(tok, eof, window, d.sync)
		return d.w.err
	}
	return nil
}

// writeBlockSkip writes the current block and uses the number of tokens
// to determine if the block should be stored on no matches, or
// only huffman encoded.
func (d *compressor) writeBlockSkip(tok *tokens, index int, eof bool) error {
	if index > 0 || eof {
		if d.blockStart <= index {
			window := d.window[d.blockStart:index]
			// If we removed less than a 64th of all literals
			// we huffman compress the block.
			if int(tok.n) > len(window)-int(tok.n>>6) {
				d.w.writeBlockHuff(eof, window, d.sync)
			} else {
				// Write a dynamic huffman block.
				d.w.writeBlockDynamic(tok, eof, window, d.sync)
			}
		} else {
			d.w.writeBlock(tok, eof, nil)
		}
		d.blockStart = index
		return d.w.err
	}
	return nil
}

// fillWindow will fill the current window with the supplied
// dictionary and calculate all hashes.
// This is much faster than doing a full encode.
// Should only be used after a start/reset.
func (d *compressor) fillWindow(b []byte) {
	// Do not fill window if we are in store-only or huffman mode.
	if d.level <= 0 {
		return
	}
	if d.fast != nil {
		// encode the last data, but discard the result
		if len(b) > maxMatchOffset {
			b = b[len(b)-maxMatchOffset:]
		}
		d.fast.Encode(&d.tokens, b)
		d.tokens.Reset()
		return
	}
	s := d.state
	// If we are given too much, cut it.
	if len(b) > windowSize {
		b = b[len(b)-windowSize:]
	}
	// Add all to window.
	n := copy(d.window[d.windowEnd:], b)

	// Calculate 256 hashes at the time (more L1 cache hits)
	loops := (n + 256 - minMatchLength) / 256
	for j := range loops {
		startindex := j * 256
		end := min(startindex+256+minMatchLength-1, n)
		tocheck := d.window[startindex:end]
		dstSize := len(tocheck) - minMatchLength + 1

		if dstSize <= 0 {
			continue
		}

		dst := s.hashMatch[:dstSize]
		bulkHash4(tocheck, dst)
		var newH uint32
		for i, val := range dst {
			di := i + startindex
			newH = val & hashMask
			// Get previous value with the same hash.
			// Our chain should point to the previous value.
			s.hashPrev[di&windowMask] = s.hashHead[newH]
			// Set the head of the hash chain to us.
			s.hashHead[newH] = uint32(di + s.hashOffset)
		}
	}
	// Update window information.
	d.windowEnd += n
	s.index = n
}

// Try to find a match starting at index whose length is greater than prevSize.
// We only look at chainCount possibilities before giving up.
func (d *compressor) findMatch(pos int, prevHead int, lookahead int) (length, offset int, ok bool) {
	minMatchLook := min(lookahead, maxMatchLength)

	win := d.window[0 : pos+minMatchLook]

	// We quit when we get a match that's at least nice long
	nice := min(d.nice, len(win)-pos)

	// If we've got a match that's good enough, only look in 1/4 the chain.
	tries := d.chain
	length = minMatchLength - 1

	wEnd := win[pos+length]
	wPos := win[pos:]
	minIndex := max(pos-windowSize, 0)
	offset = 0

	if d.chain < 100 {
		for i := prevHead; tries > 0; tries-- {
			if wEnd == win[i+length] {
				n := matchLen(win[i:i+minMatchLook], wPos)
				if n > length {
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
			if i <= minIndex {
				// hashPrev[i & windowMask] has already been overwritten, so stop now.
				break
			}
			i = int(d.state.hashPrev[i&windowMask]) - d.state.hashOffset
			if i < minIndex {
				break
			}
		}
		return
	}

	// Minimum gain to accept a match.
	cGain := 4

	// Some like it higher (CSV), some like it lower (JSON)
	const baseCost = 3
	// Base is 4 bytes at with an additional cost.
	// Matches must be better than this.

	for i := prevHead; tries > 0; tries-- {
		if wEnd == win[i+length] {
			n := matchLen(win[i:i+minMatchLook], wPos)
			if n > length {
				// Calculate gain. Estimates the gains of the new match compared to emitting as literals.
				newGain := d.h.bitLengthRaw(wPos[:n]) - int(offsetExtraBits[offsetCode(uint32(pos-i))]) - baseCost - int(lengthExtraBits[lengthCodes[(n-3)&255]])

				if newGain > cGain {
					length = n
					offset = pos - i
					cGain = newGain
					ok = true
					if n >= nice {
						// The match is good enough that we don't try to find a better one.
						break
					}
					wEnd = win[pos+n]
				}
			}
		}
		if i <= minIndex {
			// hashPrev[i & windowMask] has already been overwritten, so stop now.
			break
		}
		i = int(d.state.hashPrev[i&windowMask]) - d.state.hashOffset
		if i < minIndex {
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

// hash4 returns a hash representation of the first 4 bytes
// of the supplied slice.
// The caller must ensure that len(b) >= 4.
func hash4(b []byte) uint32 {
	return hash4u(loadLE32(b, 0), hashBits)
}

// hash4 returns the hash of u to fit in a hash table with h bits.
// Preferably h should be a constant and should always be <32.
func hash4u(u uint32, h uint8) uint32 {
	return (u * prime4bytes) >> (32 - h)
}

// bulkHash4 will compute hashes using the same
// algorithm as hash4
func bulkHash4(b []byte, dst []uint32) {
	if len(b) < 4 {
		return
	}
	hb := loadLE32(b, 0)

	dst[0] = hash4u(hb, hashBits)
	end := len(b) - 4 + 1
	for i := 1; i < end; i++ {
		hb = (hb >> 8) | uint32(b[i+3])<<24
		dst[i] = hash4u(hb, hashBits)
	}
}

func (d *compressor) initDeflate() {
	d.window = make([]byte, 2*windowSize)
	d.byteAvailable = false
	d.err = nil
	if d.state == nil {
		return
	}
	s := d.state
	s.index = 0
	s.hashOffset = 1
	s.length = minMatchLength - 1
	s.offset = 0
	s.chainHead = -1
}

// deflateLazy does encoding with lazy matching.
func (d *compressor) deflateLazy() {
	s := d.state

	if d.windowEnd-s.index < minMatchLength+maxMatchLength && !d.sync {
		return
	}
	if d.windowEnd != s.index && d.chain > 100 {
		// Get literal huffman coder.
		// This is used to estimate the cost of emitting a literal.
		if d.h == nil {
			d.h = newHuffmanEncoder(maxFlateBlockTokens)
		}
		var tmp [256]uint16
		for _, v := range d.window[s.index:d.windowEnd] {
			tmp[v]++
		}
		d.h.generate(tmp[:], 15)
	}

	s.maxInsertIndex = d.windowEnd - (minMatchLength - 1)

	for {
		lookahead := d.windowEnd - s.index
		if lookahead < minMatchLength+maxMatchLength {
			if !d.sync {
				return
			}
			if lookahead == 0 {
				// Flush current output block if any.
				if d.byteAvailable {
					// There is still one pending token that needs to be flushed
					d.tokens.AddLiteral(d.window[s.index-1])
					d.byteAvailable = false
				}
				if d.tokens.n > 0 {
					if d.err = d.writeBlock(&d.tokens, s.index, false); d.err != nil {
						return
					}
					d.tokens.Reset()
				}
				return
			}
		}
		if s.index < s.maxInsertIndex {
			// Update the hash
			hash := hash4(d.window[s.index:])
			ch := s.hashHead[hash]
			s.chainHead = int(ch)
			s.hashPrev[s.index&windowMask] = ch
			s.hashHead[hash] = uint32(s.index + s.hashOffset)
		}
		prevLength := s.length
		prevOffset := s.offset
		s.length = minMatchLength - 1
		s.offset = 0
		minIndex := max(s.index-windowSize, 0)

		if s.chainHead-s.hashOffset >= minIndex && lookahead > prevLength && prevLength < d.lazy {
			if newLength, newOffset, ok := d.findMatch(s.index, s.chainHead-s.hashOffset, lookahead); ok {
				s.length = newLength
				s.offset = newOffset
			}
		}

		if prevLength >= minMatchLength && s.length <= prevLength {
			// No better match, but check for better match at end...
			//
			// Skip forward a number of bytes.
			// Offset of 2 seems to yield the best results. 3 is sometimes better.
			const checkOff = 2

			// Check all, except full length
			if prevLength < maxMatchLength-checkOff {
				prevIndex := s.index - 1
				if prevIndex+prevLength < s.maxInsertIndex {
					end := min(lookahead, maxMatchLength+checkOff)
					end += prevIndex

					// Hash at match end.
					h := hash4(d.window[prevIndex+prevLength:])
					ch2 := int(s.hashHead[h]) - s.hashOffset - prevLength
					if prevIndex-ch2 != prevOffset && ch2 > minIndex+checkOff {
						length := matchLen(d.window[prevIndex+checkOff:end], d.window[ch2+checkOff:])
						// It seems like a pure length metric is best.
						if length > prevLength {
							prevLength = length
							prevOffset = prevIndex - ch2

							// Extend back...
							for i := checkOff - 1; i >= 0; i-- {
								if prevLength >= maxMatchLength || d.window[prevIndex+i] != d.window[ch2+i] {
									// Emit tokens we "owe"
									for j := 0; j <= i; j++ {
										d.tokens.AddLiteral(d.window[prevIndex+j])
										if d.tokens.n == maxFlateBlockTokens {
											// The block includes the current character
											if d.err = d.writeBlock(&d.tokens, s.index, false); d.err != nil {
												return
											}
											d.tokens.Reset()
										}
										s.index++
										if s.index < s.maxInsertIndex {
											h := hash4(d.window[s.index:])
											ch := s.hashHead[h]
											s.chainHead = int(ch)
											s.hashPrev[s.index&windowMask] = ch
											s.hashHead[h] = uint32(s.index + s.hashOffset)
										}
									}
									break
								} else {
									prevLength++
								}
							}
						}
					}
				}
			}
			// There was a match at the previous step, and the current match is
			// not better. Output the previous match.
			d.tokens.AddMatch(uint32(prevLength-3), uint32(prevOffset-minOffsetSize))

			// Insert in the hash table all strings up to the end of the match.
			// index and index-1 are already inserted. If there is not enough
			// lookahead, the last two strings are not inserted into the hash
			// table.
			newIndex := s.index + prevLength - 1
			// Calculate missing hashes
			end := min(newIndex, s.maxInsertIndex)
			end += minMatchLength - 1
			startindex := min(s.index+1, s.maxInsertIndex)
			tocheck := d.window[startindex:end]
			dstSize := len(tocheck) - minMatchLength + 1
			if dstSize > 0 {
				dst := s.hashMatch[:dstSize]
				bulkHash4(tocheck, dst)
				var newH uint32
				for i, val := range dst {
					di := i + startindex
					newH = val & hashMask
					// Get previous value with the same hash.
					// Our chain should point to the previous value.
					s.hashPrev[di&windowMask] = s.hashHead[newH]
					// Set the head of the hash chain to us.
					s.hashHead[newH] = uint32(di + s.hashOffset)
				}
			}

			s.index = newIndex
			d.byteAvailable = false
			s.length = minMatchLength - 1
			if d.tokens.n == maxFlateBlockTokens {
				// The block includes the current character
				if d.err = d.writeBlock(&d.tokens, s.index, false); d.err != nil {
					return
				}
				d.tokens.Reset()
			}
			s.ii = 0
		} else {
			// Reset, if we got a match this run.
			if s.length >= minMatchLength {
				s.ii = 0
			}
			// We have a byte waiting. Emit it.
			if d.byteAvailable {
				s.ii++
				d.tokens.AddLiteral(d.window[s.index-1])
				if d.tokens.n == maxFlateBlockTokens {
					if d.err = d.writeBlock(&d.tokens, s.index, false); d.err != nil {
						return
					}
					d.tokens.Reset()
				}
				s.index++

				// If we have a long run of no matches, skip additional bytes
				// Resets when s.ii overflows after 64KB.
				if n := int(s.ii) - d.chain; n > 0 {
					n = 1 + int(n>>6)
					for j := 0; j < n; j++ {
						if s.index >= d.windowEnd-1 {
							break
						}
						d.tokens.AddLiteral(d.window[s.index-1])
						if d.tokens.n == maxFlateBlockTokens {
							if d.err = d.writeBlock(&d.tokens, s.index, false); d.err != nil {
								return
							}
							d.tokens.Reset()
						}
						// Index...
						if s.index < s.maxInsertIndex {
							h := hash4(d.window[s.index:])
							ch := s.hashHead[h]
							s.chainHead = int(ch)
							s.hashPrev[s.index&windowMask] = ch
							s.hashHead[h] = uint32(s.index + s.hashOffset)
						}
						s.index++
					}
					// Flush last byte
					d.tokens.AddLiteral(d.window[s.index-1])
					d.byteAvailable = false
					// s.length = minMatchLength - 1 // not needed, since s.ii is reset above, so it should never be > minMatchLength
					if d.tokens.n == maxFlateBlockTokens {
						if d.err = d.writeBlock(&d.tokens, s.index, false); d.err != nil {
							return
						}
						d.tokens.Reset()
					}
				}
			} else {
				s.index++
				d.byteAvailable = true
			}
		}
	}
}

func (d *compressor) store() {
	if d.windowEnd > 0 && (d.windowEnd == maxStoreBlockSize || d.sync) {
		d.err = d.writeStoredBlock(d.window[:d.windowEnd])
		d.windowEnd = 0
	}
}

// fillWindow will fill the buffer with data for huffman-only compression.
// The number of bytes copied is returned.
func (d *compressor) fillBlock(b []byte) int {
	n := copy(d.window[d.windowEnd:], b)
	d.windowEnd += n
	return n
}

// storeHuff will compress and store the currently added data,
// if enough has been accumulated or we at the end of the stream.
// Any error that occurred will be in d.err
func (d *compressor) storeHuff() {
	if d.windowEnd < len(d.window) && !d.sync || d.windowEnd == 0 {
		return
	}
	d.w.writeBlockHuff(false, d.window[:d.windowEnd], d.sync)
	d.err = d.w.err
	d.windowEnd = 0
}

// storeFast will compress and store the currently added data,
// if enough has been accumulated or we at the end of the stream.
// Any error that occurred will be in d.err
func (d *compressor) storeFast() {
	// We only compress if we have maxStoreBlockSize.
	if d.windowEnd < len(d.window) {
		if !d.sync {
			return
		}
		// Handle extremely small sizes.
		if d.windowEnd < 128 {
			if d.windowEnd == 0 {
				return
			}
			if d.windowEnd <= 32 {
				d.err = d.writeStoredBlock(d.window[:d.windowEnd])
			} else {
				d.w.writeBlockHuff(false, d.window[:d.windowEnd], true)
				d.err = d.w.err
			}
			d.tokens.Reset()
			d.windowEnd = 0
			d.fast.Reset()
			return
		}
	}

	d.fast.Encode(&d.tokens, d.window[:d.windowEnd])
	// If we made zero matches, store the block as is.
	if d.tokens.n == 0 {
		d.err = d.writeStoredBlock(d.window[:d.windowEnd])
		// If we removed less than 1/16th, huffman compress the block.
	} else if int(d.tokens.n) > d.windowEnd-(d.windowEnd>>4) {
		d.w.writeBlockHuff(false, d.window[:d.windowEnd], d.sync)
		d.err = d.w.err
	} else {
		d.w.writeBlockDynamic(&d.tokens, false, d.window[:d.windowEnd], d.sync)
		d.err = d.w.err
	}
	d.tokens.Reset()
	d.windowEnd = 0
}

// write will add input byte to the stream.
// Unless an error occurs all bytes will be consumed.
func (d *compressor) write(b []byte) (n int, err error) {
	if d.err != nil {
		return 0, d.err
	}
	n = len(b)
	for len(b) > 0 {
		if d.windowEnd == len(d.window) || d.sync {
			d.step(d)
		}
		b = b[d.fill(d, b):]
		if d.err != nil {
			return 0, d.err
		}
	}
	return n, d.err
}

func (d *compressor) syncFlush() error {
	d.sync = true
	if d.err != nil {
		return d.err
	}
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
		d.fill = (*compressor).fillBlock
		d.step = (*compressor).store
	case level == HuffmanOnly:
		d.w.logNewTablePenalty = 10
		d.window = make([]byte, 32<<10)
		d.fill = (*compressor).fillBlock
		d.step = (*compressor).storeHuff
	case level == DefaultCompression:
		level = 6
		fallthrough
	case level >= 1 && level <= 6:
		d.w.logNewTablePenalty = 7
		d.fast = newFastEnc(level)
		d.window = make([]byte, maxStoreBlockSize)
		d.fill = (*compressor).fillBlock
		d.step = (*compressor).storeFast
	case 7 <= level && level <= 9:
		d.w.logNewTablePenalty = 8
		d.state = &advancedState{}
		d.compressionLevel = levels[level]
		d.initDeflate()
		d.fill = (*compressor).fillDeflate
		d.step = (*compressor).deflateLazy
	default:
		return fmt.Errorf("flate: invalid compression level %d: want value in range [-2, 9]", level)
	}
	d.level = level
	return nil
}

func (d *compressor) reset(w io.Writer) {
	d.w.reset(w)
	d.sync = false
	d.err = nil
	// We only need to reset a few things for Snappy.
	if d.fast != nil {
		d.fast.Reset()
		d.windowEnd = 0
		d.tokens.Reset()
		return
	}
	switch d.compressionLevel.chain {
	case 0:
		// level was NoCompression or ConstantCompression.
		d.windowEnd = 0
	default:
		s := d.state
		s.chainHead = -1
		for i := range s.hashHead {
			s.hashHead[i] = 0
		}
		for i := range s.hashPrev {
			s.hashPrev[i] = 0
		}
		s.hashOffset = 1
		s.index, d.windowEnd = 0, 0
		d.blockStart, d.byteAvailable = 0, false
		d.tokens.Reset()
		s.length = minMatchLength - 1
		s.offset = 0
		s.ii = 0
		s.maxInsertIndex = 0
	}
}

var errWriterClosed = errors.New("flate: closed writer")

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
	d.w.reset(nil)
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
	zw, err := NewWriter(w, level)
	if err != nil {
		return nil, err
	}
	zw.d.fillWindow(dict)
	zw.dict = append(zw.dict, dict...) // duplicate dictionary for Reset method.
	return zw, err
}

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
// the result of NewWriter or NewWriterDict called with dst
// and w's level and dictionary.
func (w *Writer) Reset(dst io.Writer) {
	if len(w.dict) > 0 {
		// w was created with NewWriterDict
		w.d.reset(dst)
		if dst != nil {
			w.d.fillWindow(w.dict)
		}
	} else {
		// w was created with NewWriter
		w.d.reset(dst)
	}
}

// ResetDict discards the writer's state and makes it equivalent to
// the result of NewWriter or NewWriterDict called with dst
// and w's level, but sets a specific dictionary.
func (w *Writer) ResetDict(dst io.Writer, dict []byte) {
	w.dict = dict
	w.d.reset(dst)
	w.d.fillWindow(w.dict)
}
