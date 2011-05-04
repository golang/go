// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package spdy is an incomplete implementation of the SPDY protocol.
//
// The implementation follows draft 2 of the spec:
// https://sites.google.com/a/chromium.org/dev/spdy/spdy-protocol/spdy-protocol-draft2
package spdy

import (
	"bytes"
	"compress/zlib"
	"encoding/binary"
	"http"
	"io"
	"os"
	"strconv"
	"strings"
	"sync"
)

// Version is the protocol version number that this package implements.
const Version = 2

// ControlFrameType stores the type field in a control frame header.
type ControlFrameType uint16

// Control frame type constants
const (
	TypeSynStream    ControlFrameType = 0x0001
	TypeSynReply     = 0x0002
	TypeRstStream    = 0x0003
	TypeSettings     = 0x0004
	TypeNoop         = 0x0005
	TypePing         = 0x0006
	TypeGoaway       = 0x0007
	TypeHeaders      = 0x0008
	TypeWindowUpdate = 0x0009
)

func (t ControlFrameType) String() string {
	switch t {
	case TypeSynStream:
		return "SYN_STREAM"
	case TypeSynReply:
		return "SYN_REPLY"
	case TypeRstStream:
		return "RST_STREAM"
	case TypeSettings:
		return "SETTINGS"
	case TypeNoop:
		return "NOOP"
	case TypePing:
		return "PING"
	case TypeGoaway:
		return "GOAWAY"
	case TypeHeaders:
		return "HEADERS"
	case TypeWindowUpdate:
		return "WINDOW_UPDATE"
	}
	return "Type(" + strconv.Itoa(int(t)) + ")"
}

type FrameFlags uint8

// Stream frame flags
const (
	FlagFin            FrameFlags = 0x01
	FlagUnidirectional = 0x02
)

// SETTINGS frame flags
const (
	FlagClearPreviouslyPersistedSettings FrameFlags = 0x01
)

// MaxDataLength is the maximum number of bytes that can be stored in one frame.
const MaxDataLength = 1<<24 - 1

// A Frame is a framed message as sent between clients and servers.
// There are two types of frames: control frames and data frames.
type Frame struct {
	Header [4]byte
	Flags  FrameFlags
	Data   []byte
}

// ControlFrame creates a control frame with the given information.
func ControlFrame(t ControlFrameType, f FrameFlags, data []byte) Frame {
	return Frame{
		Header: [4]byte{
			(Version&0xff00)>>8 | 0x80,
			(Version & 0x00ff),
			byte((t & 0xff00) >> 8),
			byte((t & 0x00ff) >> 0),
		},
		Flags: f,
		Data:  data,
	}
}

// DataFrame creates a data frame with the given information.
func DataFrame(streamId uint32, f FrameFlags, data []byte) Frame {
	return Frame{
		Header: [4]byte{
			byte(streamId & 0x7f000000 >> 24),
			byte(streamId & 0x00ff0000 >> 16),
			byte(streamId & 0x0000ff00 >> 8),
			byte(streamId & 0x000000ff >> 0),
		},
		Flags: f,
		Data:  data,
	}
}

// ReadFrame reads an entire frame into memory.
func ReadFrame(r io.Reader) (f Frame, err os.Error) {
	_, err = io.ReadFull(r, f.Header[:])
	if err != nil {
		return
	}
	err = binary.Read(r, binary.BigEndian, &f.Flags)
	if err != nil {
		return
	}
	var lengthField [3]byte
	_, err = io.ReadFull(r, lengthField[:])
	if err != nil {
		if err == os.EOF {
			err = io.ErrUnexpectedEOF
		}
		return
	}
	var length uint32
	length |= uint32(lengthField[0]) << 16
	length |= uint32(lengthField[1]) << 8
	length |= uint32(lengthField[2]) << 0
	if length > 0 {
		f.Data = make([]byte, int(length))
		_, err = io.ReadFull(r, f.Data)
		if err == os.EOF {
			err = io.ErrUnexpectedEOF
		}
	} else {
		f.Data = []byte{}
	}
	return
}

// IsControl returns whether the frame holds a control frame.
func (f Frame) IsControl() bool {
	return f.Header[0]&0x80 != 0
}

// Type obtains the type field if the frame is a control frame, otherwise it returns zero.
func (f Frame) Type() ControlFrameType {
	if !f.IsControl() {
		return 0
	}
	return (ControlFrameType(f.Header[2])<<8 | ControlFrameType(f.Header[3]))
}

// StreamId returns the stream ID field if the frame is a data frame, otherwise it returns zero.
func (f Frame) StreamId() (id uint32) {
	if f.IsControl() {
		return 0
	}
	id |= uint32(f.Header[0]) << 24
	id |= uint32(f.Header[1]) << 16
	id |= uint32(f.Header[2]) << 8
	id |= uint32(f.Header[3]) << 0
	return
}

// WriteTo writes the frame in the SPDY format.
func (f Frame) WriteTo(w io.Writer) (n int64, err os.Error) {
	var nn int
	// Header
	nn, err = w.Write(f.Header[:])
	n += int64(nn)
	if err != nil {
		return
	}
	// Flags
	nn, err = w.Write([]byte{byte(f.Flags)})
	n += int64(nn)
	if err != nil {
		return
	}
	// Length
	nn, err = w.Write([]byte{
		byte(len(f.Data) & 0x00ff0000 >> 16),
		byte(len(f.Data) & 0x0000ff00 >> 8),
		byte(len(f.Data) & 0x000000ff),
	})
	n += int64(nn)
	if err != nil {
		return
	}
	// Data
	if len(f.Data) > 0 {
		nn, err = w.Write(f.Data)
		n += int64(nn)
	}
	return
}

// headerDictionary is the dictionary sent to the zlib compressor/decompressor.
// Even though the specification states there is no null byte at the end, Chrome sends it.
const headerDictionary = "optionsgetheadpostputdeletetrace" +
	"acceptaccept-charsetaccept-encodingaccept-languageauthorizationexpectfromhost" +
	"if-modified-sinceif-matchif-none-matchif-rangeif-unmodifiedsince" +
	"max-forwardsproxy-authorizationrangerefererteuser-agent" +
	"100101200201202203204205206300301302303304305306307400401402403404405406407408409410411412413414415416417500501502503504505" +
	"accept-rangesageetaglocationproxy-authenticatepublicretry-after" +
	"servervarywarningwww-authenticateallowcontent-basecontent-encodingcache-control" +
	"connectiondatetrailertransfer-encodingupgradeviawarning" +
	"content-languagecontent-lengthcontent-locationcontent-md5content-rangecontent-typeetagexpireslast-modifiedset-cookie" +
	"MondayTuesdayWednesdayThursdayFridaySaturdaySunday" +
	"JanFebMarAprMayJunJulAugSepOctNovDec" +
	"chunkedtext/htmlimage/pngimage/jpgimage/gifapplication/xmlapplication/xhtmltext/plainpublicmax-age" +
	"charset=iso-8859-1utf-8gzipdeflateHTTP/1.1statusversionurl\x00"

// hrSource is a reader that passes through reads from another reader.
// When the underlying reader reaches EOF, Read will block until another reader is added via change.
type hrSource struct {
	r io.Reader
	m sync.RWMutex
	c *sync.Cond
}

func (src *hrSource) Read(p []byte) (n int, err os.Error) {
	src.m.RLock()
	for src.r == nil {
		src.c.Wait()
	}
	n, err = src.r.Read(p)
	src.m.RUnlock()
	if err == os.EOF {
		src.change(nil)
		err = nil
	}
	return
}

func (src *hrSource) change(r io.Reader) {
	src.m.Lock()
	defer src.m.Unlock()
	src.r = r
	src.c.Broadcast()
}

// A HeaderReader reads zlib-compressed headers.
type HeaderReader struct {
	source       hrSource
	decompressor io.ReadCloser
}

// NewHeaderReader creates a HeaderReader with the initial dictionary.
func NewHeaderReader() (hr *HeaderReader) {
	hr = new(HeaderReader)
	hr.source.c = sync.NewCond(hr.source.m.RLocker())
	return
}

// ReadHeader reads a set of headers from a reader.
func (hr *HeaderReader) ReadHeader(r io.Reader) (h http.Header, err os.Error) {
	hr.source.change(r)
	h, err = hr.read()
	return
}

// Decode reads a set of headers from a block of bytes.
func (hr *HeaderReader) Decode(data []byte) (h http.Header, err os.Error) {
	hr.source.change(bytes.NewBuffer(data))
	h, err = hr.read()
	return
}

func (hr *HeaderReader) read() (h http.Header, err os.Error) {
	var count uint16
	if hr.decompressor == nil {
		hr.decompressor, err = zlib.NewReaderDict(&hr.source, []byte(headerDictionary))
		if err != nil {
			return
		}
	}
	err = binary.Read(hr.decompressor, binary.BigEndian, &count)
	if err != nil {
		return
	}
	h = make(http.Header, int(count))
	for i := 0; i < int(count); i++ {
		var name, value string
		name, err = readHeaderString(hr.decompressor)
		if err != nil {
			return
		}
		value, err = readHeaderString(hr.decompressor)
		if err != nil {
			return
		}
		valueList := strings.Split(string(value), "\x00", -1)
		for _, v := range valueList {
			h.Add(name, v)
		}
	}
	return
}

func readHeaderString(r io.Reader) (s string, err os.Error) {
	var length uint16
	err = binary.Read(r, binary.BigEndian, &length)
	if err != nil {
		return
	}
	data := make([]byte, int(length))
	_, err = io.ReadFull(r, data)
	if err != nil {
		return
	}
	return string(data), nil
}

// HeaderWriter will write zlib-compressed headers on different streams.
type HeaderWriter struct {
	compressor *zlib.Writer
	buffer     *bytes.Buffer
}

// NewHeaderWriter creates a HeaderWriter ready to compress headers.
func NewHeaderWriter(level int) (hw *HeaderWriter) {
	hw = &HeaderWriter{buffer: new(bytes.Buffer)}
	hw.compressor, _ = zlib.NewWriterDict(hw.buffer, level, []byte(headerDictionary))
	return
}

// WriteHeader writes a header block directly to an output.
func (hw *HeaderWriter) WriteHeader(w io.Writer, h http.Header) (err os.Error) {
	hw.write(h)
	_, err = io.Copy(w, hw.buffer)
	hw.buffer.Reset()
	return
}

// Encode returns a compressed header block.
func (hw *HeaderWriter) Encode(h http.Header) (data []byte) {
	hw.write(h)
	data = make([]byte, hw.buffer.Len())
	hw.buffer.Read(data)
	return
}

func (hw *HeaderWriter) write(h http.Header) {
	binary.Write(hw.compressor, binary.BigEndian, uint16(len(h)))
	for k, vals := range h {
		k = strings.ToLower(k)
		binary.Write(hw.compressor, binary.BigEndian, uint16(len(k)))
		binary.Write(hw.compressor, binary.BigEndian, []byte(k))
		v := strings.Join(vals, "\x00")
		binary.Write(hw.compressor, binary.BigEndian, uint16(len(v)))
		binary.Write(hw.compressor, binary.BigEndian, []byte(v))
	}
	hw.compressor.Flush()
}
