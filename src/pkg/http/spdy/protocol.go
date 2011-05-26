// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package spdy is an incomplete implementation of the SPDY protocol.
//
// The implementation follows draft 2 of the spec:
// https://sites.google.com/a/chromium.org/dev/spdy/spdy-protocol/spdy-protocol-draft2
package spdy

import (
	"encoding/binary"
	"http"
	"os"
)

//  Data Frame Format
//  +----------------------------------+
//  |0|       Stream-ID (31bits)       |
//  +----------------------------------+
//  | flags (8)  |  Length (24 bits)   |
//  +----------------------------------+
//  |               Data               |
//  +----------------------------------+
//
//  Control Frame Format
//  +----------------------------------+
//  |1| Version(15bits) | Type(16bits) |
//  +----------------------------------+
//  | flags (8)  |  Length (24 bits)   |
//  +----------------------------------+
//  |               Data               |
//  +----------------------------------+
//
//  Control Frame: SYN_STREAM
//  +----------------------------------+
//  |1|000000000000001|0000000000000001|
//  +----------------------------------+
//  | flags (8)  |  Length (24 bits)   |  >= 12
//  +----------------------------------+
//  |X|       Stream-ID(31bits)        |
//  +----------------------------------+
//  |X|Associated-To-Stream-ID (31bits)|
//  +----------------------------------+
//  |Pri| unused      | Length (16bits)|
//  +----------------------------------+
//
//  Control Frame: SYN_REPLY
//  +----------------------------------+
//  |1|000000000000001|0000000000000010|
//  +----------------------------------+
//  | flags (8)  |  Length (24 bits)   |  >= 8
//  +----------------------------------+
//  |X|       Stream-ID(31bits)        |
//  +----------------------------------+
//  | unused (16 bits)| Length (16bits)|
//  +----------------------------------+
//
//  Control Frame: RST_STREAM
//  +----------------------------------+
//  |1|000000000000001|0000000000000011|
//  +----------------------------------+
//  | flags (8)  |  Length (24 bits)   |  >= 4
//  +----------------------------------+
//  |X|       Stream-ID(31bits)        |
//  +----------------------------------+
//  |        Status code (32 bits)     |
//  +----------------------------------+
//
//  Control Frame: SETTINGS
//  +----------------------------------+
//  |1|000000000000001|0000000000000100|
//  +----------------------------------+
//  | flags (8)  |  Length (24 bits)   |
//  +----------------------------------+
//  |        # of entries (32)         |
//  +----------------------------------+
//
//  Control Frame: NOOP
//  +----------------------------------+
//  |1|000000000000001|0000000000000101|
//  +----------------------------------+
//  | flags (8)  |  Length (24 bits)   | = 0
//  +----------------------------------+
//
//  Control Frame: PING
//  +----------------------------------+
//  |1|000000000000001|0000000000000110|
//  +----------------------------------+
//  | flags (8)  |  Length (24 bits)   | = 4
//  +----------------------------------+
//  |        Unique id (32 bits)       |
//  +----------------------------------+
//
//  Control Frame: GOAWAY
//  +----------------------------------+
//  |1|000000000000001|0000000000000111|
//  +----------------------------------+
//  | flags (8)  |  Length (24 bits)   | = 4
//  +----------------------------------+
//  |X|  Last-accepted-stream-id       |
//  +----------------------------------+
//
//  Control Frame: HEADERS
//  +----------------------------------+
//  |1|000000000000001|0000000000001000|
//  +----------------------------------+
//  | flags (8)  |  Length (24 bits)   | >= 8
//  +----------------------------------+
//  |X|      Stream-ID (31 bits)       |
//  +----------------------------------+
//  | unused (16 bits)| Length (16bits)|
//  +----------------------------------+
//
//  Control Frame: WINDOW_UPDATE
//  +----------------------------------+
//  |1|000000000000001|0000000000001001|
//  +----------------------------------+
//  | flags (8)  |  Length (24 bits)   | = 8
//  +----------------------------------+
//  |X|      Stream-ID (31 bits)       |
//  +----------------------------------+
//  |   Delta-Window-Size (32 bits)    |
//  +----------------------------------+

// Version is the protocol version number that this package implements.
const Version = 2

// ControlFrameType stores the type field in a control frame header.
type ControlFrameType uint16

// Control frame type constants
const (
	TypeSynStream    ControlFrameType = 0x0001
	TypeSynReply                      = 0x0002
	TypeRstStream                     = 0x0003
	TypeSettings                      = 0x0004
	TypeNoop                          = 0x0005
	TypePing                          = 0x0006
	TypeGoAway                        = 0x0007
	TypeHeaders                       = 0x0008
	TypeWindowUpdate                  = 0x0009
)

// ControlFlags are the flags that can be set on a control frame.
type ControlFlags uint8

const (
	ControlFlagFin ControlFlags = 0x01
)

// DataFlags are the flags that can be set on a data frame.
type DataFlags uint8

const (
	DataFlagFin        DataFlags = 0x01
	DataFlagCompressed           = 0x02
)

// MaxDataLength is the maximum number of bytes that can be stored in one frame.
const MaxDataLength = 1<<24 - 1

// Frame is a single SPDY frame in its unpacked in-memory representation. Use
// Framer to read and write it.
type Frame interface {
	write(f *Framer) os.Error
}

// ControlFrameHeader contains all the fields in a control frame header,
// in its unpacked in-memory representation.
type ControlFrameHeader struct {
	// Note, high bit is the "Control" bit.
	version   uint16
	frameType ControlFrameType
	Flags     ControlFlags
	length    uint32
}

type controlFrame interface {
	Frame
	read(h ControlFrameHeader, f *Framer) os.Error
}

// SynStreamFrame is the unpacked, in-memory representation of a SYN_STREAM
// frame.
type SynStreamFrame struct {
	CFHeader             ControlFrameHeader
	StreamId             uint32
	AssociatedToStreamId uint32
	// Note, only 2 highest bits currently used
	// Rest of Priority is unused.
	Priority uint16
	Headers  http.Header
}

func (frame *SynStreamFrame) write(f *Framer) os.Error {
	return f.writeSynStreamFrame(frame)
}

func (frame *SynStreamFrame) read(h ControlFrameHeader, f *Framer) os.Error {
	return f.readSynStreamFrame(h, frame)
}

// SynReplyFrame is the unpacked, in-memory representation of a SYN_REPLY frame.
type SynReplyFrame struct {
	CFHeader ControlFrameHeader
	StreamId uint32
	Headers  http.Header
}

func (frame *SynReplyFrame) write(f *Framer) os.Error {
	return f.writeSynReplyFrame(frame)
}

func (frame *SynReplyFrame) read(h ControlFrameHeader, f *Framer) os.Error {
	return f.readSynReplyFrame(h, frame)
}

// StatusCode represents the status that led to a RST_STREAM
type StatusCode uint32

const (
	ProtocolError      StatusCode = 1
	InvalidStream                 = 2
	RefusedStream                 = 3
	UnsupportedVersion            = 4
	Cancel                        = 5
	InternalError                 = 6
	FlowControlError              = 7
)

// RstStreamFrame is the unpacked, in-memory representation of a RST_STREAM
// frame.
type RstStreamFrame struct {
	CFHeader ControlFrameHeader
	StreamId uint32
	Status   StatusCode
}

func (frame *RstStreamFrame) write(f *Framer) (err os.Error) {
	frame.CFHeader.version = Version
	frame.CFHeader.frameType = TypeRstStream
	frame.CFHeader.length = 8

	// Serialize frame to Writer
	if err = writeControlFrameHeader(f.w, frame.CFHeader); err != nil {
		return
	}
	if err = binary.Write(f.w, binary.BigEndian, frame.StreamId); err != nil {
		return
	}
	if err = binary.Write(f.w, binary.BigEndian, frame.Status); err != nil {
		return
	}
	return
}

func (frame *RstStreamFrame) read(h ControlFrameHeader, f *Framer) os.Error {
	frame.CFHeader = h
	if err := binary.Read(f.r, binary.BigEndian, &frame.StreamId); err != nil {
		return err
	}
	if err := binary.Read(f.r, binary.BigEndian, &frame.Status); err != nil {
		return err
	}
	return nil
}

// SettingsFlag represents a flag in a SETTINGS frame.
type SettingsFlag uint8

const (
	FlagSettingsPersistValue SettingsFlag = 0x1
	FlagSettingsPersisted                 = 0x2
)

// SettingsFlag represents the id of an id/value pair in a SETTINGS frame.
type SettingsId uint32

const (
	SettingsUploadBandwidth      SettingsId = 1
	SettingsDownloadBandwidth               = 2
	SettingsRoundTripTime                   = 3
	SettingsMaxConcurrentStreams            = 4
	SettingsCurrentCwnd                     = 5
)

// SettingsFlagIdValue is the unpacked, in-memory representation of the
// combined flag/id/value for a setting in a SETTINGS frame.
type SettingsFlagIdValue struct {
	Flag  SettingsFlag
	Id    SettingsId
	Value uint32
}

// SettingsFrame is the unpacked, in-memory representation of a SPDY
// SETTINGS frame.
type SettingsFrame struct {
	CFHeader     ControlFrameHeader
	FlagIdValues []SettingsFlagIdValue
}

func (frame *SettingsFrame) write(f *Framer) (err os.Error) {
	frame.CFHeader.version = Version
	frame.CFHeader.frameType = TypeSettings
	frame.CFHeader.length = uint32(len(frame.FlagIdValues)*8 + 4)

	// Serialize frame to Writer
	if err = writeControlFrameHeader(f.w, frame.CFHeader); err != nil {
		return
	}
	if err = binary.Write(f.w, binary.BigEndian, uint32(len(frame.FlagIdValues))); err != nil {
		return
	}
	for _, flagIdValue := range frame.FlagIdValues {
		flagId := (uint32(flagIdValue.Flag) << 24) | uint32(flagIdValue.Id)
		if err = binary.Write(f.w, binary.BigEndian, flagId); err != nil {
			return
		}
		if err = binary.Write(f.w, binary.BigEndian, flagIdValue.Value); err != nil {
			return
		}
	}
	return
}

func (frame *SettingsFrame) read(h ControlFrameHeader, f *Framer) os.Error {
	frame.CFHeader = h
	var numSettings uint32
	if err := binary.Read(f.r, binary.BigEndian, &numSettings); err != nil {
		return err
	}
	frame.FlagIdValues = make([]SettingsFlagIdValue, numSettings)
	for i := uint32(0); i < numSettings; i++ {
		if err := binary.Read(f.r, binary.BigEndian, &frame.FlagIdValues[i].Id); err != nil {
			return err
		}
		frame.FlagIdValues[i].Flag = SettingsFlag((frame.FlagIdValues[i].Id & 0xff000000) >> 24)
		frame.FlagIdValues[i].Id &= 0xffffff
		if err := binary.Read(f.r, binary.BigEndian, &frame.FlagIdValues[i].Value); err != nil {
			return err
		}
	}
	return nil
}

// NoopFrame is the unpacked, in-memory representation of a NOOP frame.
type NoopFrame struct {
	CFHeader ControlFrameHeader
}

func (frame *NoopFrame) write(f *Framer) os.Error {
	frame.CFHeader.version = Version
	frame.CFHeader.frameType = TypeNoop

	// Serialize frame to Writer
	return writeControlFrameHeader(f.w, frame.CFHeader)
}

func (frame *NoopFrame) read(h ControlFrameHeader, f *Framer) os.Error {
	frame.CFHeader = h
	return nil
}

// PingFrame is the unpacked, in-memory representation of a PING frame.
type PingFrame struct {
	CFHeader ControlFrameHeader
	Id       uint32
}

func (frame *PingFrame) write(f *Framer) (err os.Error) {
	frame.CFHeader.version = Version
	frame.CFHeader.frameType = TypePing
	frame.CFHeader.length = 4

	// Serialize frame to Writer
	if err = writeControlFrameHeader(f.w, frame.CFHeader); err != nil {
		return
	}
	if err = binary.Write(f.w, binary.BigEndian, frame.Id); err != nil {
		return
	}
	return
}

func (frame *PingFrame) read(h ControlFrameHeader, f *Framer) os.Error {
	frame.CFHeader = h
	if err := binary.Read(f.r, binary.BigEndian, &frame.Id); err != nil {
		return err
	}
	return nil
}

// GoAwayFrame is the unpacked, in-memory representation of a GOAWAY frame.
type GoAwayFrame struct {
	CFHeader         ControlFrameHeader
	LastGoodStreamId uint32
}

func (frame *GoAwayFrame) write(f *Framer) (err os.Error) {
	frame.CFHeader.version = Version
	frame.CFHeader.frameType = TypeGoAway
	frame.CFHeader.length = 4

	// Serialize frame to Writer
	if err = writeControlFrameHeader(f.w, frame.CFHeader); err != nil {
		return
	}
	if err = binary.Write(f.w, binary.BigEndian, frame.LastGoodStreamId); err != nil {
		return
	}
	return nil
}

func (frame *GoAwayFrame) read(h ControlFrameHeader, f *Framer) os.Error {
	frame.CFHeader = h
	if err := binary.Read(f.r, binary.BigEndian, &frame.LastGoodStreamId); err != nil {
		return err
	}
	return nil
}

// HeadersFrame is the unpacked, in-memory representation of a HEADERS frame.
type HeadersFrame struct {
	CFHeader ControlFrameHeader
	StreamId uint32
	Headers  http.Header
}

func (frame *HeadersFrame) write(f *Framer) os.Error {
	return f.writeHeadersFrame(frame)
}

func (frame *HeadersFrame) read(h ControlFrameHeader, f *Framer) os.Error {
	return f.readHeadersFrame(h, frame)
}

func newControlFrame(frameType ControlFrameType) (controlFrame, os.Error) {
	ctor, ok := cframeCtor[frameType]
	if !ok {
		return nil, InvalidControlFrame
	}
	return ctor(), nil
}

var cframeCtor = map[ControlFrameType]func() controlFrame{
	TypeSynStream: func() controlFrame { return new(SynStreamFrame) },
	TypeSynReply:  func() controlFrame { return new(SynReplyFrame) },
	TypeRstStream: func() controlFrame { return new(RstStreamFrame) },
	TypeSettings:  func() controlFrame { return new(SettingsFrame) },
	TypeNoop:      func() controlFrame { return new(NoopFrame) },
	TypePing:      func() controlFrame { return new(PingFrame) },
	TypeGoAway:    func() controlFrame { return new(GoAwayFrame) },
	TypeHeaders:   func() controlFrame { return new(HeadersFrame) },
	// TODO(willchan): Add TypeWindowUpdate
}

// DataFrame is the unpacked, in-memory representation of a DATA frame.
type DataFrame struct {
	// Note, high bit is the "Control" bit. Should be 0 for data frames.
	StreamId uint32
	Flags    DataFlags
	Data     []byte
}

func (frame *DataFrame) write(f *Framer) os.Error {
	return f.writeDataFrame(frame)
}

// HeaderDictionary is the dictionary sent to the zlib compressor/decompressor.
// Even though the specification states there is no null byte at the end, Chrome sends it.
const HeaderDictionary = "optionsgetheadpostputdeletetrace" +
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
