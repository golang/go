// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package spdy

import (
	"compress/zlib"
	"encoding/binary"
	"io"
	"net/http"
	"strings"
)

func (frame *SynStreamFrame) read(h ControlFrameHeader, f *Framer) error {
	return f.readSynStreamFrame(h, frame)
}

func (frame *SynReplyFrame) read(h ControlFrameHeader, f *Framer) error {
	return f.readSynReplyFrame(h, frame)
}

func (frame *RstStreamFrame) read(h ControlFrameHeader, f *Framer) error {
	frame.CFHeader = h
	if err := binary.Read(f.r, binary.BigEndian, &frame.StreamId); err != nil {
		return err
	}
	if err := binary.Read(f.r, binary.BigEndian, &frame.Status); err != nil {
		return err
	}
	return nil
}

func (frame *SettingsFrame) read(h ControlFrameHeader, f *Framer) error {
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

func (frame *NoopFrame) read(h ControlFrameHeader, f *Framer) error {
	frame.CFHeader = h
	return nil
}

func (frame *PingFrame) read(h ControlFrameHeader, f *Framer) error {
	frame.CFHeader = h
	if err := binary.Read(f.r, binary.BigEndian, &frame.Id); err != nil {
		return err
	}
	return nil
}

func (frame *GoAwayFrame) read(h ControlFrameHeader, f *Framer) error {
	frame.CFHeader = h
	if err := binary.Read(f.r, binary.BigEndian, &frame.LastGoodStreamId); err != nil {
		return err
	}
	return nil
}

func (frame *HeadersFrame) read(h ControlFrameHeader, f *Framer) error {
	return f.readHeadersFrame(h, frame)
}

func newControlFrame(frameType ControlFrameType) (controlFrame, error) {
	ctor, ok := cframeCtor[frameType]
	if !ok {
		return nil, &Error{Err: InvalidControlFrame}
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

func (f *Framer) uncorkHeaderDecompressor(payloadSize int64) error {
	if f.headerDecompressor != nil {
		f.headerReader.N = payloadSize
		return nil
	}
	f.headerReader = io.LimitedReader{R: f.r, N: payloadSize}
	decompressor, err := zlib.NewReaderDict(&f.headerReader, []byte(HeaderDictionary))
	if err != nil {
		return err
	}
	f.headerDecompressor = decompressor
	return nil
}

// ReadFrame reads SPDY encoded data and returns a decompressed Frame.
func (f *Framer) ReadFrame() (Frame, error) {
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

func (f *Framer) parseControlFrame(version uint16, frameType ControlFrameType) (Frame, error) {
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

func parseHeaderValueBlock(r io.Reader, streamId uint32) (http.Header, error) {
	var numHeaders uint16
	if err := binary.Read(r, binary.BigEndian, &numHeaders); err != nil {
		return nil, err
	}
	var e error
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
			e = &Error{UnlowercasedHeaderName, streamId}
			name = strings.ToLower(name)
		}
		if h[name] != nil {
			e = &Error{DuplicateHeaders, streamId}
		}
		if err := binary.Read(r, binary.BigEndian, &length); err != nil {
			return nil, err
		}
		value := make([]byte, length)
		if _, err := io.ReadFull(r, value); err != nil {
			return nil, err
		}
		valueList := strings.Split(string(value), "\x00")
		for _, v := range valueList {
			h.Add(name, v)
		}
	}
	if e != nil {
		return h, e
	}
	return h, nil
}

func (f *Framer) readSynStreamFrame(h ControlFrameHeader, frame *SynStreamFrame) error {
	frame.CFHeader = h
	var err error
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
		f.uncorkHeaderDecompressor(int64(h.length - 10))
		reader = f.headerDecompressor
	}

	frame.Headers, err = parseHeaderValueBlock(reader, frame.StreamId)
	if !f.headerCompressionDisabled && ((err == io.EOF && f.headerReader.N == 0) || f.headerReader.N != 0) {
		err = &Error{WrongCompressedPayloadSize, 0}
	}
	if err != nil {
		return err
	}
	// Remove this condition when we bump Version to 3.
	if Version >= 3 {
		for h := range frame.Headers {
			if invalidReqHeaders[h] {
				return &Error{InvalidHeaderPresent, frame.StreamId}
			}
		}
	}
	return nil
}

func (f *Framer) readSynReplyFrame(h ControlFrameHeader, frame *SynReplyFrame) error {
	frame.CFHeader = h
	var err error
	if err = binary.Read(f.r, binary.BigEndian, &frame.StreamId); err != nil {
		return err
	}
	var unused uint16
	if err = binary.Read(f.r, binary.BigEndian, &unused); err != nil {
		return err
	}
	reader := f.r
	if !f.headerCompressionDisabled {
		f.uncorkHeaderDecompressor(int64(h.length - 6))
		reader = f.headerDecompressor
	}
	frame.Headers, err = parseHeaderValueBlock(reader, frame.StreamId)
	if !f.headerCompressionDisabled && ((err == io.EOF && f.headerReader.N == 0) || f.headerReader.N != 0) {
		err = &Error{WrongCompressedPayloadSize, 0}
	}
	if err != nil {
		return err
	}
	// Remove this condition when we bump Version to 3.
	if Version >= 3 {
		for h := range frame.Headers {
			if invalidRespHeaders[h] {
				return &Error{InvalidHeaderPresent, frame.StreamId}
			}
		}
	}
	return nil
}

func (f *Framer) readHeadersFrame(h ControlFrameHeader, frame *HeadersFrame) error {
	frame.CFHeader = h
	var err error
	if err = binary.Read(f.r, binary.BigEndian, &frame.StreamId); err != nil {
		return err
	}
	var unused uint16
	if err = binary.Read(f.r, binary.BigEndian, &unused); err != nil {
		return err
	}
	reader := f.r
	if !f.headerCompressionDisabled {
		f.uncorkHeaderDecompressor(int64(h.length - 6))
		reader = f.headerDecompressor
	}
	frame.Headers, err = parseHeaderValueBlock(reader, frame.StreamId)
	if !f.headerCompressionDisabled && ((err == io.EOF && f.headerReader.N == 0) || f.headerReader.N != 0) {
		err = &Error{WrongCompressedPayloadSize, 0}
	}
	if err != nil {
		return err
	}

	// Remove this condition when we bump Version to 3.
	if Version >= 3 {
		var invalidHeaders map[string]bool
		if frame.StreamId%2 == 0 {
			invalidHeaders = invalidReqHeaders
		} else {
			invalidHeaders = invalidRespHeaders
		}
		for h := range frame.Headers {
			if invalidHeaders[h] {
				return &Error{InvalidHeaderPresent, frame.StreamId}
			}
		}
	}
	return nil
}

func (f *Framer) parseDataFrame(streamId uint32) (*DataFrame, error) {
	var length uint32
	if err := binary.Read(f.r, binary.BigEndian, &length); err != nil {
		return nil, err
	}
	var frame DataFrame
	frame.StreamId = streamId
	frame.Flags = DataFlags(length >> 24)
	length &= 0xffffff
	frame.Data = make([]byte, length)
	if _, err := io.ReadFull(f.r, frame.Data); err != nil {
		return nil, err
	}
	return &frame, nil
}
