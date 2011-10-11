// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package spdy

import (
	"bytes"
	"http"
	"io"
	"reflect"
	"testing"
)

func TestHeaderParsing(t *testing.T) {
	headers := http.Header{
		"Url":     []string{"http://www.google.com/"},
		"Method":  []string{"get"},
		"Version": []string{"http/1.1"},
	}
	var headerValueBlockBuf bytes.Buffer
	writeHeaderValueBlock(&headerValueBlockBuf, headers)

	const bogusStreamId = 1
	newHeaders, err := parseHeaderValueBlock(&headerValueBlockBuf, bogusStreamId)
	if err != nil {
		t.Fatal("parseHeaderValueBlock:", err)
	}

	if !reflect.DeepEqual(headers, newHeaders) {
		t.Fatal("got: ", newHeaders, "\nwant: ", headers)
	}
}

func TestCreateParseSynStreamFrame(t *testing.T) {
	buffer := new(bytes.Buffer)
	framer := &Framer{
		headerCompressionDisabled: true,
		w:                         buffer,
		headerBuf:                 new(bytes.Buffer),
		r:                         buffer,
	}
	synStreamFrame := SynStreamFrame{
		CFHeader: ControlFrameHeader{
			version:   Version,
			frameType: TypeSynStream,
		},
		Headers: http.Header{
			"Url":     []string{"http://www.google.com/"},
			"Method":  []string{"get"},
			"Version": []string{"http/1.1"},
		},
	}
	if err := framer.WriteFrame(&synStreamFrame); err != nil {
		t.Fatal("WriteFrame without compression:", err)
	}
	frame, err := framer.ReadFrame()
	if err != nil {
		t.Fatal("ReadFrame without compression:", err)
	}
	parsedSynStreamFrame, ok := frame.(*SynStreamFrame)
	if !ok {
		t.Fatal("Parsed incorrect frame type:", frame)
	}
	if !reflect.DeepEqual(synStreamFrame, *parsedSynStreamFrame) {
		t.Fatal("got: ", *parsedSynStreamFrame, "\nwant: ", synStreamFrame)
	}

	// Test again with compression
	buffer.Reset()
	framer, err = NewFramer(buffer, buffer)
	if err != nil {
		t.Fatal("Failed to create new framer:", err)
	}
	if err := framer.WriteFrame(&synStreamFrame); err != nil {
		t.Fatal("WriteFrame with compression:", err)
	}
	frame, err = framer.ReadFrame()
	if err != nil {
		t.Fatal("ReadFrame with compression:", err)
	}
	parsedSynStreamFrame, ok = frame.(*SynStreamFrame)
	if !ok {
		t.Fatal("Parsed incorrect frame type:", frame)
	}
	if !reflect.DeepEqual(synStreamFrame, *parsedSynStreamFrame) {
		t.Fatal("got: ", *parsedSynStreamFrame, "\nwant: ", synStreamFrame)
	}
}

func TestCreateParseSynReplyFrame(t *testing.T) {
	buffer := new(bytes.Buffer)
	framer := &Framer{
		headerCompressionDisabled: true,
		w:                         buffer,
		headerBuf:                 new(bytes.Buffer),
		r:                         buffer,
	}
	synReplyFrame := SynReplyFrame{
		CFHeader: ControlFrameHeader{
			version:   Version,
			frameType: TypeSynReply,
		},
		Headers: http.Header{
			"Url":     []string{"http://www.google.com/"},
			"Method":  []string{"get"},
			"Version": []string{"http/1.1"},
		},
	}
	if err := framer.WriteFrame(&synReplyFrame); err != nil {
		t.Fatal("WriteFrame without compression:", err)
	}
	frame, err := framer.ReadFrame()
	if err != nil {
		t.Fatal("ReadFrame without compression:", err)
	}
	parsedSynReplyFrame, ok := frame.(*SynReplyFrame)
	if !ok {
		t.Fatal("Parsed incorrect frame type:", frame)
	}
	if !reflect.DeepEqual(synReplyFrame, *parsedSynReplyFrame) {
		t.Fatal("got: ", *parsedSynReplyFrame, "\nwant: ", synReplyFrame)
	}

	// Test again with compression
	buffer.Reset()
	framer, err = NewFramer(buffer, buffer)
	if err != nil {
		t.Fatal("Failed to create new framer:", err)
	}
	if err := framer.WriteFrame(&synReplyFrame); err != nil {
		t.Fatal("WriteFrame with compression:", err)
	}
	frame, err = framer.ReadFrame()
	if err != nil {
		t.Fatal("ReadFrame with compression:", err)
	}
	parsedSynReplyFrame, ok = frame.(*SynReplyFrame)
	if !ok {
		t.Fatal("Parsed incorrect frame type:", frame)
	}
	if !reflect.DeepEqual(synReplyFrame, *parsedSynReplyFrame) {
		t.Fatal("got: ", *parsedSynReplyFrame, "\nwant: ", synReplyFrame)
	}
}

func TestCreateParseRstStream(t *testing.T) {
	buffer := new(bytes.Buffer)
	framer, err := NewFramer(buffer, buffer)
	if err != nil {
		t.Fatal("Failed to create new framer:", err)
	}
	rstStreamFrame := RstStreamFrame{
		CFHeader: ControlFrameHeader{
			version:   Version,
			frameType: TypeRstStream,
		},
		StreamId: 1,
		Status:   InvalidStream,
	}
	if err := framer.WriteFrame(&rstStreamFrame); err != nil {
		t.Fatal("WriteFrame:", err)
	}
	frame, err := framer.ReadFrame()
	if err != nil {
		t.Fatal("ReadFrame:", err)
	}
	parsedRstStreamFrame, ok := frame.(*RstStreamFrame)
	if !ok {
		t.Fatal("Parsed incorrect frame type:", frame)
	}
	if !reflect.DeepEqual(rstStreamFrame, *parsedRstStreamFrame) {
		t.Fatal("got: ", *parsedRstStreamFrame, "\nwant: ", rstStreamFrame)
	}
}

func TestCreateParseSettings(t *testing.T) {
	buffer := new(bytes.Buffer)
	framer, err := NewFramer(buffer, buffer)
	if err != nil {
		t.Fatal("Failed to create new framer:", err)
	}
	settingsFrame := SettingsFrame{
		CFHeader: ControlFrameHeader{
			version:   Version,
			frameType: TypeSettings,
		},
		FlagIdValues: []SettingsFlagIdValue{
			{FlagSettingsPersistValue, SettingsCurrentCwnd, 10},
			{FlagSettingsPersisted, SettingsUploadBandwidth, 1},
		},
	}
	if err := framer.WriteFrame(&settingsFrame); err != nil {
		t.Fatal("WriteFrame:", err)
	}
	frame, err := framer.ReadFrame()
	if err != nil {
		t.Fatal("ReadFrame:", err)
	}
	parsedSettingsFrame, ok := frame.(*SettingsFrame)
	if !ok {
		t.Fatal("Parsed incorrect frame type:", frame)
	}
	if !reflect.DeepEqual(settingsFrame, *parsedSettingsFrame) {
		t.Fatal("got: ", *parsedSettingsFrame, "\nwant: ", settingsFrame)
	}
}

func TestCreateParseNoop(t *testing.T) {
	buffer := new(bytes.Buffer)
	framer, err := NewFramer(buffer, buffer)
	if err != nil {
		t.Fatal("Failed to create new framer:", err)
	}
	noopFrame := NoopFrame{
		CFHeader: ControlFrameHeader{
			version:   Version,
			frameType: TypeNoop,
		},
	}
	if err := framer.WriteFrame(&noopFrame); err != nil {
		t.Fatal("WriteFrame:", err)
	}
	frame, err := framer.ReadFrame()
	if err != nil {
		t.Fatal("ReadFrame:", err)
	}
	parsedNoopFrame, ok := frame.(*NoopFrame)
	if !ok {
		t.Fatal("Parsed incorrect frame type:", frame)
	}
	if !reflect.DeepEqual(noopFrame, *parsedNoopFrame) {
		t.Fatal("got: ", *parsedNoopFrame, "\nwant: ", noopFrame)
	}
}

func TestCreateParsePing(t *testing.T) {
	buffer := new(bytes.Buffer)
	framer, err := NewFramer(buffer, buffer)
	if err != nil {
		t.Fatal("Failed to create new framer:", err)
	}
	pingFrame := PingFrame{
		CFHeader: ControlFrameHeader{
			version:   Version,
			frameType: TypePing,
		},
		Id: 31337,
	}
	if err := framer.WriteFrame(&pingFrame); err != nil {
		t.Fatal("WriteFrame:", err)
	}
	frame, err := framer.ReadFrame()
	if err != nil {
		t.Fatal("ReadFrame:", err)
	}
	parsedPingFrame, ok := frame.(*PingFrame)
	if !ok {
		t.Fatal("Parsed incorrect frame type:", frame)
	}
	if !reflect.DeepEqual(pingFrame, *parsedPingFrame) {
		t.Fatal("got: ", *parsedPingFrame, "\nwant: ", pingFrame)
	}
}

func TestCreateParseGoAway(t *testing.T) {
	buffer := new(bytes.Buffer)
	framer, err := NewFramer(buffer, buffer)
	if err != nil {
		t.Fatal("Failed to create new framer:", err)
	}
	goAwayFrame := GoAwayFrame{
		CFHeader: ControlFrameHeader{
			version:   Version,
			frameType: TypeGoAway,
		},
		LastGoodStreamId: 31337,
	}
	if err := framer.WriteFrame(&goAwayFrame); err != nil {
		t.Fatal("WriteFrame:", err)
	}
	frame, err := framer.ReadFrame()
	if err != nil {
		t.Fatal("ReadFrame:", err)
	}
	parsedGoAwayFrame, ok := frame.(*GoAwayFrame)
	if !ok {
		t.Fatal("Parsed incorrect frame type:", frame)
	}
	if !reflect.DeepEqual(goAwayFrame, *parsedGoAwayFrame) {
		t.Fatal("got: ", *parsedGoAwayFrame, "\nwant: ", goAwayFrame)
	}
}

func TestCreateParseHeadersFrame(t *testing.T) {
	buffer := new(bytes.Buffer)
	framer := &Framer{
		headerCompressionDisabled: true,
		w:                         buffer,
		headerBuf:                 new(bytes.Buffer),
		r:                         buffer,
	}
	headersFrame := HeadersFrame{
		CFHeader: ControlFrameHeader{
			version:   Version,
			frameType: TypeHeaders,
		},
	}
	headersFrame.Headers = http.Header{
		"Url":     []string{"http://www.google.com/"},
		"Method":  []string{"get"},
		"Version": []string{"http/1.1"},
	}
	if err := framer.WriteFrame(&headersFrame); err != nil {
		t.Fatal("WriteFrame without compression:", err)
	}
	frame, err := framer.ReadFrame()
	if err != nil {
		t.Fatal("ReadFrame without compression:", err)
	}
	parsedHeadersFrame, ok := frame.(*HeadersFrame)
	if !ok {
		t.Fatal("Parsed incorrect frame type:", frame)
	}
	if !reflect.DeepEqual(headersFrame, *parsedHeadersFrame) {
		t.Fatal("got: ", *parsedHeadersFrame, "\nwant: ", headersFrame)
	}

	// Test again with compression
	buffer.Reset()
	framer, err = NewFramer(buffer, buffer)
	if err := framer.WriteFrame(&headersFrame); err != nil {
		t.Fatal("WriteFrame with compression:", err)
	}
	frame, err = framer.ReadFrame()
	if err != nil {
		t.Fatal("ReadFrame with compression:", err)
	}
	parsedHeadersFrame, ok = frame.(*HeadersFrame)
	if !ok {
		t.Fatal("Parsed incorrect frame type:", frame)
	}
	if !reflect.DeepEqual(headersFrame, *parsedHeadersFrame) {
		t.Fatal("got: ", *parsedHeadersFrame, "\nwant: ", headersFrame)
	}
}

func TestCreateParseDataFrame(t *testing.T) {
	buffer := new(bytes.Buffer)
	framer, err := NewFramer(buffer, buffer)
	if err != nil {
		t.Fatal("Failed to create new framer:", err)
	}
	dataFrame := DataFrame{
		StreamId: 1,
		Data:     []byte{'h', 'e', 'l', 'l', 'o'},
	}
	if err := framer.WriteFrame(&dataFrame); err != nil {
		t.Fatal("WriteFrame:", err)
	}
	frame, err := framer.ReadFrame()
	if err != nil {
		t.Fatal("ReadFrame:", err)
	}
	parsedDataFrame, ok := frame.(*DataFrame)
	if !ok {
		t.Fatal("Parsed incorrect frame type:", frame)
	}
	if !reflect.DeepEqual(dataFrame, *parsedDataFrame) {
		t.Fatal("got: ", *parsedDataFrame, "\nwant: ", dataFrame)
	}
}

func TestCompressionContextAcrossFrames(t *testing.T) {
	buffer := new(bytes.Buffer)
	framer, err := NewFramer(buffer, buffer)
	if err != nil {
		t.Fatal("Failed to create new framer:", err)
	}
	headersFrame := HeadersFrame{
		CFHeader: ControlFrameHeader{
			version:   Version,
			frameType: TypeHeaders,
		},
		Headers: http.Header{
			"Url":     []string{"http://www.google.com/"},
			"Method":  []string{"get"},
			"Version": []string{"http/1.1"},
		},
	}
	if err := framer.WriteFrame(&headersFrame); err != nil {
		t.Fatal("WriteFrame (HEADERS):", err)
	}
	synStreamFrame := SynStreamFrame{ControlFrameHeader{Version, TypeSynStream, 0, 0}, 0, 0, 0, nil}
	synStreamFrame.Headers = http.Header{
		"Url":     []string{"http://www.google.com/"},
		"Method":  []string{"get"},
		"Version": []string{"http/1.1"},
	}
	if err := framer.WriteFrame(&synStreamFrame); err != nil {
		t.Fatal("WriteFrame (SYN_STREAM):", err)
	}
	frame, err := framer.ReadFrame()
	if err != nil {
		t.Fatal("ReadFrame (HEADERS):", err, buffer.Bytes())
	}
	parsedHeadersFrame, ok := frame.(*HeadersFrame)
	if !ok {
		t.Fatalf("expected HeadersFrame; got %T %v", frame, frame)
	}
	if !reflect.DeepEqual(headersFrame, *parsedHeadersFrame) {
		t.Fatal("got: ", *parsedHeadersFrame, "\nwant: ", headersFrame)
	}
	frame, err = framer.ReadFrame()
	if err != nil {
		t.Fatal("ReadFrame (SYN_STREAM):", err, buffer.Bytes())
	}
	parsedSynStreamFrame, ok := frame.(*SynStreamFrame)
	if !ok {
		t.Fatalf("expected SynStreamFrame; got %T %v", frame, frame)
	}
	if !reflect.DeepEqual(synStreamFrame, *parsedSynStreamFrame) {
		t.Fatal("got: ", *parsedSynStreamFrame, "\nwant: ", synStreamFrame)
	}
}

func TestMultipleSPDYFrames(t *testing.T) {
	// Initialize the framers.
	pr1, pw1 := io.Pipe()
	pr2, pw2 := io.Pipe()
	writer, err := NewFramer(pw1, pr2)
	if err != nil {
		t.Fatal("Failed to create writer:", err)
	}
	reader, err := NewFramer(pw2, pr1)
	if err != nil {
		t.Fatal("Failed to create reader:", err)
	}

	// Set up the frames we're actually transferring.
	headersFrame := HeadersFrame{
		CFHeader: ControlFrameHeader{
			version:   Version,
			frameType: TypeHeaders,
		},
		Headers: http.Header{
			"Url":     []string{"http://www.google.com/"},
			"Method":  []string{"get"},
			"Version": []string{"http/1.1"},
		},
	}
	synStreamFrame := SynStreamFrame{
		CFHeader: ControlFrameHeader{
			version:   Version,
			frameType: TypeSynStream,
		},
		Headers: http.Header{
			"Url":     []string{"http://www.google.com/"},
			"Method":  []string{"get"},
			"Version": []string{"http/1.1"},
		},
	}

	// Start the goroutines to write the frames.
	go func() {
		if err := writer.WriteFrame(&headersFrame); err != nil {
			t.Fatal("WriteFrame (HEADERS): ", err)
		}
		if err := writer.WriteFrame(&synStreamFrame); err != nil {
			t.Fatal("WriteFrame (SYN_STREAM): ", err)
		}
	}()

	// Read the frames and verify they look as expected.
	frame, err := reader.ReadFrame()
	if err != nil {
		t.Fatal("ReadFrame (HEADERS): ", err)
	}
	parsedHeadersFrame, ok := frame.(*HeadersFrame)
	if !ok {
		t.Fatal("Parsed incorrect frame type:", frame)
	}
	if !reflect.DeepEqual(headersFrame, *parsedHeadersFrame) {
		t.Fatal("got: ", *parsedHeadersFrame, "\nwant: ", headersFrame)
	}
	frame, err = reader.ReadFrame()
	if err != nil {
		t.Fatal("ReadFrame (SYN_STREAM):", err)
	}
	parsedSynStreamFrame, ok := frame.(*SynStreamFrame)
	if !ok {
		t.Fatal("Parsed incorrect frame type.")
	}
	if !reflect.DeepEqual(synStreamFrame, *parsedSynStreamFrame) {
		t.Fatal("got: ", *parsedSynStreamFrame, "\nwant: ", synStreamFrame)
	}
}
