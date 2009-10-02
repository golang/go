// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flate

import (
	"bytes";
	"io";
	"math";
	"os";
)

const (
	NoCompression = 0;
	BestSpeed = 1;
	fastCompression = 3;
	BestCompression = 9;
	DefaultCompression = -1;

	logMaxOffsetSize = 15; // Standard DEFLATE
	wideLogMaxOffsetSize = 22; // Wide DEFLATE
	minMatchLength = 3; // The smallest match that the deflater looks for
	maxMatchLength = 258; // The longest match for the deflater
	minOffsetSize = 1; // The shortest offset that makes any sence

	// The maximum number of tokens we put into a single flat block, just too
	// stop things from getting too large.
	maxFlateBlockTokens = 1 << 14;
	maxStoreBlockSize = 65535;
	hashBits = 15;
	hashSize = 1 << hashBits;
	hashMask = (1 << hashBits) - 1;
	hashShift = (hashBits + minMatchLength - 1) / minMatchLength;
)

type syncPipeReader struct {
	*io.PipeReader;
	closeChan chan bool;
}

func (sr *syncPipeReader) CloseWithError(err os.Error) os.Error {
	retErr := sr.PipeReader.CloseWithError(err);
	sr.closeChan <- true; // finish writer close
	return retErr;
}

type syncPipeWriter struct {
	*io.PipeWriter;
	closeChan chan bool;
}

type compressionLevel struct {
	good, lazy, nice, chain, fastSkipHashing int;
}

var levels = [] compressionLevel {
	compressionLevel {}, // 0
	// For levels 1-3 we don't bother trying with lazy matches
	compressionLevel { 3, 0, 8, 4, 4, },
	compressionLevel { 3, 0, 16, 8, 5, },
	compressionLevel { 3, 0, 32, 32, 6 },
	// Levels 4-9 use increasingly more lazy matching
	// and increasingly stringent conditions for "good enough".
	compressionLevel { 4, 4, 16, 16, math.MaxInt32 },
	compressionLevel { 8, 16, 32, 32, math.MaxInt32 },
	compressionLevel { 8, 16, 128, 128, math.MaxInt32 },
	compressionLevel { 8, 32, 128, 256, math.MaxInt32 },
	compressionLevel { 32, 128, 258, 1024, math.MaxInt32 },
	compressionLevel { 32, 258, 258, 4096, math.MaxInt32 },
}

func (sw *syncPipeWriter) Close() os.Error {
	err := sw.PipeWriter.Close();
	<-sw.closeChan; // wait for reader close
	return err;
}

func syncPipe() (*syncPipeReader, *syncPipeWriter) {
	r, w := io.Pipe();
	sr := &syncPipeReader{r, make(chan bool, 1)};
	sw := &syncPipeWriter{w, sr.closeChan};
	return sr, sw;
}

type deflater struct {
	level int;
	logWindowSize uint;
	w *huffmanBitWriter;
	r io.Reader;
	// (1 << logWindowSize) - 1.
	windowMask int;

	// hashHead[hashValue] contains the largest inputIndex with the specified hash value
	hashHead []int;

	// If hashHead[hashValue] is within the current window, then
	// hashPrev[hashHead[hashValue] & windowMask] contains the previous index
	// with the same hash value.
	hashPrev []int;

	// If we find a match of length >= niceMatch, then we don't bother searching
	// any further.
	niceMatch int;

	// If we find a match of length >= goodMatch, we only do a half-hearted
	// effort at doing lazy matching starting at the next character
	goodMatch int;

	// The maximum number of chains we look at when finding a match
	maxChainLength int;

	// The sliding window we use for matching
	window []byte;

	// The index just past the last valid character
	windowEnd int;

	// index in "window" at which current block starts
	blockStart int;
}

func (d *deflater) flush() os.Error {
	d.w.flush();
	return d.w.err;
}

func (d *deflater) fillWindow(index int) (int, os.Error) {
	wSize := d.windowMask + 1;
	if index >= wSize + wSize - (minMatchLength + maxMatchLength) {
		// shift the window by wSize
		bytes.Copy(d.window, d.window[wSize:2*wSize]);
		index -= wSize;
		d.windowEnd -= wSize;
		if d.blockStart >= wSize {
			d.blockStart -= wSize;
		} else {
			d.blockStart = math.MaxInt32;
		}
		for i, h := range d.hashHead {
			d.hashHead[i] = max(h - wSize, -1);
		}
		for i, h := range d.hashPrev {
			d.hashPrev[i] = max(h - wSize, -1);
		}
	}
	var count int;
	var err os.Error;
	count, err = io.ReadAtLeast(d.r, d.window[d.windowEnd : len(d.window)], 1);
	d.windowEnd += count;
	if err == os.EOF {
		return index, nil;
	}
	return index, err;
}

func (d *deflater) writeBlock(tokens []token, index int, eof bool) os.Error {
	if index > 0 || eof {
		var window []byte;
		if d.blockStart <= index {
			window = d.window[d.blockStart:index];
		}
		d.blockStart = index;
		d.w.writeBlock(tokens, eof, window);
		return d.w.err;
	}
	return nil;
}

// Try to find a match starting at index whose length is greater than prevSize.
// We only look at chainCount possibilities before giving up.
func (d *deflater) findMatch(pos int, prevHead int, prevLength int, lookahead int) (length, offset int, ok bool) {
	win := d.window[0:pos+min(maxMatchLength, lookahead)];

	// We quit when we get a match that's at least nice long
	nice := min(d.niceMatch, len(win) - pos);

	// If we've got a match that's good enough, only look in 1/4 the chain.
	tries := d.maxChainLength;
	length = prevLength;
	if length >= d.goodMatch {
		tries >>= 2;
	}

	w0 := win[pos];
	w1 := win[pos + 1];
	wEnd := win[pos + length];
	minIndex := pos - (d.windowMask + 1);

	for i := prevHead; tries > 0; tries-- {
		if w0 == win[i] && w1 == win[i+1] && wEnd == win[i+length] {
			// The hash function ensures that if win[i] and win[i+1] match, win[i+2] matches

			n := 3;
			for pos + n < len(win) && win[i+n] == win[pos+n] {
				n++;
			}
			if n > length && (n > 3 || pos-i <= 4096) {
				length = n;
				offset = pos - i;
				ok = true;
				if n >= nice {
					// The match is good enough that we don't try to find a better one.
					break;
				}
				wEnd = win[pos+n];
			}
		}
		if i == minIndex {
			// hashPrev[i & windowMask] has already been overwritten, so stop now.
			break;
		}
		if i = d.hashPrev[i & d.windowMask]; i < minIndex || i < 0 {
			break;
		}
	}
	return;
}

func (d *deflater) writeStoredBlock(buf []byte) os.Error {
	if d.w.writeStoredHeader(len(buf), false); d.w.err != nil {
		return d.w.err;
	}
	d.w.writeBytes(buf);
	return d.w.err;
}

func (d *deflater) storedDeflate() os.Error {
	buf := make([]byte, maxStoreBlockSize);
	for {
		n, err := d.r.Read(buf);
		if n > 0 {
			if err := d.writeStoredBlock(buf[0:n]); err != nil {
				return err;
			}
		}
		if err != nil {
			if err == os.EOF {
				break;
			}
			return err;
		}
	}
	return nil;
}

func (d *deflater) doDeflate() (err os.Error) {
	// init
	d.windowMask = 1<<d.logWindowSize - 1;
	d.hashHead = make([]int, hashSize);
	d.hashPrev = make([]int, 1 << d.logWindowSize);
	d.window = make([]byte, 2 << d.logWindowSize);
	fillInts(d.hashHead, -1);
	tokens := make([]token, maxFlateBlockTokens, maxFlateBlockTokens + 1);
	l := levels[d.level];
	d.goodMatch = l.good;
	d.niceMatch = l.nice;
	d.maxChainLength = l.chain;
	lazyMatch := l.lazy;
	length := minMatchLength - 1;
	offset := 0;
	byteAvailable := false;
	isFastDeflate := l.fastSkipHashing != 0;
	index := 0;
	// run
	if index, err = d.fillWindow(index); err != nil {
		return;
	}
	maxOffset := d.windowMask + 1; // (1 << logWindowSize);
	// only need to change when you refill the window
	windowEnd := d.windowEnd;
	maxInsertIndex := windowEnd - (minMatchLength - 1);
	ti := 0;

	hash := int(0);
	if index < maxInsertIndex {
		hash = int(d.window[index])<<hashShift + int(d.window[index+1]);
	}
	chainHead := -1;
	for {
		if index > windowEnd {
			panic("index > windowEnd");
		}
		lookahead := windowEnd - index;
		if lookahead < minMatchLength + maxMatchLength {
			if index, err = d.fillWindow(index); err != nil {
				return;
			}
			windowEnd = d.windowEnd;
			if index > windowEnd {
				panic("index > windowEnd");
			}
			maxInsertIndex = windowEnd - (minMatchLength - 1);
			lookahead = windowEnd - index;
			if lookahead == 0 {
				break;
			}
		}
		if index < maxInsertIndex {
			// Update the hash
			hash = (hash<<hashShift + int(d.window[index+2])) & hashMask;
			chainHead = d.hashHead[hash];
			d.hashPrev[index & d.windowMask] = chainHead;
			d.hashHead[hash] = index;
		}
		prevLength := length;
		prevOffset := offset;
		minIndex := max(index - maxOffset, 0);
		length = minMatchLength - 1;
		offset = 0;

		if chainHead >= minIndex &&
			(isFastDeflate && lookahead > minMatchLength - 1 ||
			!isFastDeflate && lookahead > prevLength && prevLength < lazyMatch) {
			if newLength, newOffset, ok := d.findMatch(index, chainHead, minMatchLength -1 , lookahead); ok {
				length = newLength;
				offset = newOffset;
			}
		}
		if isFastDeflate && length >= minMatchLength ||
			!isFastDeflate && prevLength >= minMatchLength && length <= prevLength {
			// There was a match at the previous step, and the current match is
			// not better. Output the previous match.
			if isFastDeflate {
				tokens[ti] = matchToken(uint32(length - minMatchLength), uint32(offset - minOffsetSize));
			} else {
				tokens[ti] = matchToken(uint32(prevLength - minMatchLength), uint32(prevOffset - minOffsetSize));
			}
			ti++;
			// Insert in the hash table all strings up to the end of the match.
			// index and index-1 are already inserted. If there is not enough
			// lookahead, the last two strings are not inserted into the hash
			// table.
			if length <= l.fastSkipHashing {
				var newIndex int;
				if isFastDeflate {
					newIndex = index + length;
				} else {
					newIndex = prevLength - 1;
				}
				for index++; index < newIndex; index++ {
					if index < maxInsertIndex {
						hash = (hash<<hashShift + int(d.window[index+2])) & hashMask;
						// Get previous value with the same hash.
						// Our chain should point to the previous value.
						d.hashPrev[index & d.windowMask] = d.hashHead[hash];
						// Set the head of the hash chain to us.
						d.hashHead[hash] = index;
					}
				}
				if !isFastDeflate {
					byteAvailable = false;
					length = minMatchLength - 1;
				}
			} else {
				// For matches this long, we don't bother inserting each individual
				// item into the table.
				index += length;
				hash = (int(d.window[index])<<hashShift + int(d.window[index+1]));
			}
			if ti == maxFlateBlockTokens {
				// The block includes the current character
				if err = d.writeBlock(tokens, index, false); err != nil {
					return;
				}
				ti = 0;
			}
		} else {
			if isFastDeflate || byteAvailable {
				i := index - 1;
				if isFastDeflate {
					i = index;
				}
				tokens[ti] = literalToken(uint32(d.window[i]) & 0xFF);
				ti++;
				if ti == maxFlateBlockTokens {
					if err = d.writeBlock(tokens, i+1, false); err != nil {
						return;
					}
					ti = 0;
				}
			}
			index++;
			if !isFastDeflate {
				byteAvailable = true;
			}
		}

	}
	if byteAvailable {
		// There is still one pending token that needs to be flushed
		tokens[ti] = literalToken(uint32(d.window[index - 1]) & 0xFF);
		ti++;
	}

	if ti > 0 {
		if err = d.writeBlock(tokens[0:ti], index, false); err != nil {
			return;
		}
	}
	return;
}

func (d *deflater) deflater(r io.Reader, w io.Writer, level int, logWindowSize uint) (err os.Error) {
	d.r = r;
	d.w = newHuffmanBitWriter(w);
	d.level = level;
	d.logWindowSize = logWindowSize;

	switch {
	case level == NoCompression:
		err = d.storedDeflate();
	case level == DefaultCompression:
		d.level = 6;
		fallthrough;
	case 1 <= level && level <= 9:
		err = d.doDeflate();
	default:
		return WrongValueError { "level", 0, 9, int32(level) };
	}

	if err != nil {
		return err;
	}
	if d.w.writeStoredHeader(0, true); d.w.err != nil {
		return d.w.err;
	}
	return d.flush();
}

func newDeflater(w io.Writer, level int, logWindowSize uint) io.WriteCloser {
	var d deflater;
	pr, pw := syncPipe();
	go func() {
		err := d.deflater(pr, w, level, logWindowSize);
		pr.CloseWithError(err);
	}();
	return pw;
}

func NewDeflater(w io.Writer, level int) io.WriteCloser {
	return newDeflater(w, level, logMaxOffsetSize);
}
