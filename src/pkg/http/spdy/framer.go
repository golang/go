// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package spdy

import (
	"bytes"
	"encoding/binary"
	"compress/zlib"
	"http"
	"io"
	"os"
	"strconv"
	"strings"
)

type FramerError int

const (
	Internal FramerError = iota
	InvalidControlFrame
	UnlowercasedHeaderName
	DuplicateHeaders
	UnknownFrameType
	InvalidDataFrame
)

func (e FramerError) String() string {
	switch e {
	case Internal:
		return "Internal"
	case InvalidControlFrame:
		return "InvalidControlFrame"
	case UnlowercasedHeaderName:
		return "UnlowercasedHeaderName"
	case DuplicateHeaders:
		return "DuplicateHeaders"
	case UnknownFrameType:
		return "UnknownFrameType"
	case InvalidDataFrame:
		return "InvalidDataFrame"
	}
	return "Error(" + strconv.Itoa(int(e)) + ")"
}

// Framer handles serializing/deserializing SPDY frames, including compressing/
// decompressing payloads.
type Framer struct {
	headerCompressionDisabled bool
	w                         io.Writer
	headerBuf                 *bytes.Buffer
	headerCompressor          *zlib.Writer
	r                         io.Reader
	headerDecompressor        io.ReadCloser
}

// NewFramer allocates a new Framer for a given SPDY connection, repesented by
// a io.Writer and io.Reader. Note that Framer will read and write individual fields 
// from/to the Reader and Writer, so the caller should pass in an appropriately 
// buffered implementation to optimize performance.
func NewFramer(w io.Writer, r io.Reader) (*Framer, os.Error) {
	compressBuf := new(bytes.Buffer)
	compressor, err := zlib.NewWriterDict(compressBuf, zlib.BestCompression, []byte(HeaderDictionary))
	if err != nil {
		return nil, err
	}
	framer := &Framer{
		w:                w,
		headerBuf:        compressBuf,
		headerCompressor: compressor,
		r:                r,
	}
	return framer, nil
}

func (f *Framer) initHeaderDecompression() os.Error {
	if f.headerDecompressor != nil {
		return nil
	}
	decompressor, err := zlib.NewReaderDict(f.r, []byte(HeaderDictionary))
	if err != nil {
		return err
	}
	f.headerDecompressor = decompressor
	return nil
}

// ReadFrame reads SPDY encoded data and returns a decompressed Frame.
func (f *Framer) ReadFrame() (Frame, os.Error) {
	var firstWord uint32
	if err := binary.Read(f.r, binary.BigEndian, &firstWord); err != nil {
		return nil, err
	}
	if (firstWord & 0x80000000) != 0 {
		frameType := ControlFrameType(firstWord & 0xffff)
		version := uint16(0x7fff & (firstWord >> 16))
		return f.parseControlFrame(version, frameType)
	}
	return f.parseDataFrame(firstWord & 0x7fffffff)
}

func (f *Framer) parseControlFrame(version uint16, frameType ControlFrameType) (Frame, os.Error) {
	var length uint32
	if err := binary.Read(f.r, binary.BigEndian, &length); err != nil {
		return nil, err
	}
	flags := ControlFlags((length & 0xff000000) >> 24)
	length &= 0xffffff
	header := ControlFrameHeader{version, frameType, flags, length}
	cframe, err := newControlFrame(frameType)
	if err != nil {
		return nil, err
	}
	if err = cframe.read(header, f); err != nil {
		return nil, err
	}
	return cframe, nil
}

func parseHeaderValueBlock(r io.Reader) (http.Header, os.Error) {
	var numHeaders uint16
	if err := binary.Read(r, binary.BigEndian, &numHeaders); err != nil {
		return nil, err
	}
	h := make(http.Header, int(numHeaders))
	for i := 0; i < int(numHeaders); i++ {
		var length uint16
		if err := binary.Read(r, binary.BigEndian, &length); err != nil {
			return nil, err
		}
		nameBytes := make([]byte, length)
		if _, err := io.ReadFull(r, nameBytes); err != nil {
			return nil, err
		}
		name := string(nameBytes)
		if name != strings.ToLower(name) {
			return nil, UnlowercasedHeaderName
		}
		if h[name] != nil {
			return nil, DuplicateHeaders
		}
		if err := binary.Read(r, binary.BigEndian, &length); err != nil {
			return nil, err
		}
		value := make([]byte, length)
		if _, err := io.ReadFull(r, value); err != nil {
			return nil, err
		}
		valueList := strings.Split(string(value), "\x00", -1)
		for _, v := range valueList {
			h.Add(name, v)
		}
	}
	return h, nil
}

func (f *Framer) readSynStreamFrame(h ControlFrameHeader, frame *SynStreamFrame) os.Error {
	frame.CFHeader = h
	var err os.Error
	if err = binary.Read(f.r, binary.BigEndian, &frame.StreamId); err != nil {
		return err
	}
	if err = binary.Read(f.r, binary.BigEndian, &frame.AssociatedToStreamId); err != nil {
		return err
	}
	if err = binary.Read(f.r, binary.BigEndian, &frame.Priority); err != nil {
		return err
	}
	frame.Priority >>= 14

	reader := f.r
	if !f.headerCompressionDisabled {
		f.initHeaderDecompression()
		reader = f.headerDecompressor
	}

	frame.Headers, err = parseHeaderValueBlock(reader)
	if err != nil {
		return err
	}
	return nil
}

func (f *Framer) readSynReplyFrame(h ControlFrameHeader, frame *SynReplyFrame) os.Error {
	frame.CFHeader = h
	var err os.Error
	if err = binary.Read(f.r, binary.BigEndian, &frame.StreamId); err != nil {
		return err
	}
	var unused uint16
	if err = binary.Read(f.r, binary.BigEndian, &unused); err != nil {
		return err
	}
	reader := f.r
	if !f.headerCompressionDisabled {
		f.initHeaderDecompression()
		reader = f.headerDecompressor
	}
	frame.Headers, err = parseHeaderValueBlock(reader)
	if err != nil {
		return err
	}
	return nil
}

func (f *Framer) readHeadersFrame(h ControlFrameHeader, frame *HeadersFrame) os.Error {
	frame.CFHeader = h
	var err os.Error
	if err = binary.Read(f.r, binary.BigEndian, &frame.StreamId); err != nil {
		return err
	}
	var unused uint16
	if err = binary.Read(f.r, binary.BigEndian, &unused); err != nil {
		return err
	}
	reader := f.r
	if !f.headerCompressionDisabled {
		f.initHeaderDecompression()
		reader = f.headerDecompressor
	}
	frame.Headers, err = parseHeaderValueBlock(reader)
	if err != nil {
		return err
	}
	return nil
}

func (f *Framer) parseDataFrame(streamId uint32) (*DataFrame, os.Error) {
	var length uint32
	if err := binary.Read(f.r, binary.BigEndian, &length); err != nil {
		return nil, err
	}
	var frame DataFrame
	frame.StreamId = streamId
	frame.Flags = DataFlags(length >> 24)
	length &= 0xffffff
	frame.Data = make([]byte, length)
	// TODO(willchan): Support compressed data frames.
	if _, err := io.ReadFull(f.r, frame.Data); err != nil {
		return nil, err
	}
	return &frame, nil
}

// WriteFrame writes a frame.
func (f *Framer) WriteFrame(frame Frame) os.Error {
	return frame.write(f)
}

func writeControlFrameHeader(w io.Writer, h ControlFrameHeader) os.Error {
	if err := binary.Write(w, binary.BigEndian, 0x8000|h.version); err != nil {
		return err
	}
	if err := binary.Write(w, binary.BigEndian, h.frameType); err != nil {
		return err
	}
	flagsAndLength := (uint32(h.Flags) << 24) | h.length
	if err := binary.Write(w, binary.BigEndian, flagsAndLength); err != nil {
		return err
	}
	return nil
}

func writeHeaderValueBlock(w io.Writer, h http.Header) (n int, err os.Error) {
	n = 0
	if err = binary.Write(w, binary.BigEndian, uint16(len(h))); err != nil {
		return
	}
	n += 2
	for name, values := range h {
		if err = binary.Write(w, binary.BigEndian, uint16(len(name))); err != nil {
			return
		}
		n += 2
		name = strings.ToLower(name)
		if _, err = io.WriteString(w, name); err != nil {
			return
		}
		n += len(name)
		v := strings.Join(values, "\x00")
		if err = binary.Write(w, binary.BigEndian, uint16(len(v))); err != nil {
			return
		}
		n += 2
		if _, err = io.WriteString(w, v); err != nil {
			return
		}
		n += len(v)
	}
	return
}

func (f *Framer) writeSynStreamFrame(frame *SynStreamFrame) (err os.Error) {
	// Marshal the headers.
	var writer io.Writer = f.headerBuf
	if !f.headerCompressionDisabled {
		writer = f.headerCompressor
	}
	if _, err = writeHeaderValueBlock(writer, frame.Headers); err != nil {
		return
	}
	if !f.headerCompressionDisabled {
		f.headerCompressor.Flush()
	}

	// Set ControlFrameHeader
	frame.CFHeader.version = Version
	frame.CFHeader.frameType = TypeSynStream
	frame.CFHeader.length = uint32(len(f.headerBuf.Bytes()) + 10)

	// Serialize frame to Writer
	if err = writeControlFrameHeader(f.w, frame.CFHeader); err != nil {
		return err
	}
	if err = binary.Write(f.w, binary.BigEndian, frame.StreamId); err != nil {
		return err
	}
	if err = binary.Write(f.w, binary.BigEndian, frame.AssociatedToStreamId); err != nil {
		return err
	}
	if err = binary.Write(f.w, binary.BigEndian, frame.Priority<<14); err != nil {
		return err
	}
	if _, err = f.w.Write(f.headerBuf.Bytes()); err != nil {
		return err
	}
	f.headerBuf.Reset()
	return nil
}

func (f *Framer) writeSynReplyFrame(frame *SynReplyFrame) (err os.Error) {
	// Marshal the headers.
	var writer io.Writer = f.headerBuf
	if !f.headerCompressionDisabled {
		writer = f.headerCompressor
	}
	if _, err = writeHeaderValueBlock(writer, frame.Headers); err != nil {
		return
	}
	if !f.headerCompressionDisabled {
		f.headerCompressor.Flush()
	}

	// Set ControlFrameHeader
	frame.CFHeader.version = Version
	frame.CFHeader.frameType = TypeSynReply
	frame.CFHeader.length = uint32(len(f.headerBuf.Bytes()) + 6)

	// Serialize frame to Writer
	if err = writeControlFrameHeader(f.w, frame.CFHeader); err != nil {
		return
	}
	if err = binary.Write(f.w, binary.BigEndian, frame.StreamId); err != nil {
		return
	}
	if err = binary.Write(f.w, binary.BigEndian, uint16(0)); err != nil {
		return
	}
	if _, err = f.w.Write(f.headerBuf.Bytes()); err != nil {
		return
	}
	f.headerBuf.Reset()
	return
}

func (f *Framer) writeHeadersFrame(frame *HeadersFrame) (err os.Error) {
	// Marshal the headers.
	var writer io.Writer = f.headerBuf
	if !f.headerCompressionDisabled {
		writer = f.headerCompressor
	}
	if _, err = writeHeaderValueBlock(writer, frame.Headers); err != nil {
		return
	}
	if !f.headerCompressionDisabled {
		f.headerCompressor.Flush()
	}

	// Set ControlFrameHeader
	frame.CFHeader.version = Version
	frame.CFHeader.frameType = TypeHeaders
	frame.CFHeader.length = uint32(len(f.headerBuf.Bytes()) + 6)

	// Serialize frame to Writer
	if err = writeControlFrameHeader(f.w, frame.CFHeader); err != nil {
		return
	}
	if err = binary.Write(f.w, binary.BigEndian, frame.StreamId); err != nil {
		return
	}
	if err = binary.Write(f.w, binary.BigEndian, uint16(0)); err != nil {
		return
	}
	if _, err = f.w.Write(f.headerBuf.Bytes()); err != nil {
		return
	}
	f.headerBuf.Reset()
	return
}

func (f *Framer) writeDataFrame(frame *DataFrame) (err os.Error) {
	// Validate DataFrame
	if frame.StreamId&0x80000000 != 0 || len(frame.Data) >= 0x0f000000 {
		return InvalidDataFrame
	}

	// TODO(willchan): Support data compression.
	// Serialize frame to Writer
	if err = binary.Write(f.w, binary.BigEndian, frame.StreamId); err != nil {
		return
	}
	flagsAndLength := (uint32(frame.Flags) << 24) | uint32(len(frame.Data))
	if err = binary.Write(f.w, binary.BigEndian, flagsAndLength); err != nil {
		return
	}
	if _, err = f.w.Write(frame.Data); err != nil {
		return
	}

	return nil
}
