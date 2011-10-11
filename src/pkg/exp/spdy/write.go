// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package spdy

import (
	"encoding/binary"
	"http"
	"io"
	"os"
	"strings"
)

func (frame *SynStreamFrame) write(f *Framer) os.Error {
	return f.writeSynStreamFrame(frame)
}

func (frame *SynReplyFrame) write(f *Framer) os.Error {
	return f.writeSynReplyFrame(frame)
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

func (frame *NoopFrame) write(f *Framer) os.Error {
	frame.CFHeader.version = Version
	frame.CFHeader.frameType = TypeNoop

	// Serialize frame to Writer
	return writeControlFrameHeader(f.w, frame.CFHeader)
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

func (frame *HeadersFrame) write(f *Framer) os.Error {
	return f.writeHeadersFrame(frame)
}

func (frame *DataFrame) write(f *Framer) os.Error {
	return f.writeDataFrame(frame)
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
		return &Error{InvalidDataFrame, frame.StreamId}
	}

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
