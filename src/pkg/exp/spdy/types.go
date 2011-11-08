// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package spdy

import (
	"bytes"
	"compress/zlib"
	"io"
	"net/http"
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
	write(f *Framer) error
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
	read(h ControlFrameHeader, f *Framer) error
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

// SynReplyFrame is the unpacked, in-memory representation of a SYN_REPLY frame.
type SynReplyFrame struct {
	CFHeader ControlFrameHeader
	StreamId uint32
	Headers  http.Header
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

// NoopFrame is the unpacked, in-memory representation of a NOOP frame.
type NoopFrame struct {
	CFHeader ControlFrameHeader
}

// PingFrame is the unpacked, in-memory representation of a PING frame.
type PingFrame struct {
	CFHeader ControlFrameHeader
	Id       uint32
}

// GoAwayFrame is the unpacked, in-memory representation of a GOAWAY frame.
type GoAwayFrame struct {
	CFHeader         ControlFrameHeader
	LastGoodStreamId uint32
}

// HeadersFrame is the unpacked, in-memory representation of a HEADERS frame.
type HeadersFrame struct {
	CFHeader ControlFrameHeader
	StreamId uint32
	Headers  http.Header
}

// DataFrame is the unpacked, in-memory representation of a DATA frame.
type DataFrame struct {
	// Note, high bit is the "Control" bit. Should be 0 for data frames.
	StreamId uint32
	Flags    DataFlags
	Data     []byte
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

// A SPDY specific error.
type ErrorCode string

const (
	UnlowercasedHeaderName     ErrorCode = "header was not lowercased"
	DuplicateHeaders           ErrorCode = "multiple headers with same name"
	WrongCompressedPayloadSize ErrorCode = "compressed payload size was incorrect"
	UnknownFrameType           ErrorCode = "unknown frame type"
	InvalidControlFrame        ErrorCode = "invalid control frame"
	InvalidDataFrame           ErrorCode = "invalid data frame"
	InvalidHeaderPresent       ErrorCode = "frame contained invalid header"
)

// Error contains both the type of error and additional values. StreamId is 0
// if Error is not associated with a stream.
type Error struct {
	Err      ErrorCode
	StreamId uint32
}

func (e *Error) Error() string {
	return string(e.Err)
}

var invalidReqHeaders = map[string]bool{
	"Connection":        true,
	"Keep-Alive":        true,
	"Proxy-Connection":  true,
	"Transfer-Encoding": true,
}

var invalidRespHeaders = map[string]bool{
	"Connection":        true,
	"Keep-Alive":        true,
	"Transfer-Encoding": true,
}

// Framer handles serializing/deserializing SPDY frames, including compressing/
// decompressing payloads.
type Framer struct {
	headerCompressionDisabled bool
	w                         io.Writer
	headerBuf                 *bytes.Buffer
	headerCompressor          *zlib.Writer
	r                         io.Reader
	headerReader              io.LimitedReader
	headerDecompressor        io.ReadCloser
}

// NewFramer allocates a new Framer for a given SPDY connection, repesented by
// a io.Writer and io.Reader. Note that Framer will read and write individual fields 
// from/to the Reader and Writer, so the caller should pass in an appropriately 
// buffered implementation to optimize performance.
func NewFramer(w io.Writer, r io.Reader) (*Framer, error) {
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
