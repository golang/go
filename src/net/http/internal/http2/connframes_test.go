// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http2_test

import (
	"bytes"
	"io"
	"net/http"
	"os"
	"reflect"
	"slices"
	"testing"

	. "net/http/internal/http2"

	"golang.org/x/net/http2/hpack"
)

type testConnFramer struct {
	t   testing.TB
	fr  *Framer
	dec *hpack.Decoder
}

// readFrame reads the next frame.
// It returns nil if the conn is closed or no frames are available.
func (tf *testConnFramer) readFrame() Frame {
	tf.t.Helper()
	fr, err := tf.fr.ReadFrame()
	if err == io.EOF || err == os.ErrDeadlineExceeded {
		return nil
	}
	if err != nil {
		tf.t.Fatalf("ReadFrame: %v", err)
	}
	return fr
}

type readFramer interface {
	readFrame() Frame
}

// readFrame reads a frame of a specific type.
func readFrame[T any](t testing.TB, framer readFramer) T {
	t.Helper()
	var v T
	fr := framer.readFrame()
	if fr == nil {
		t.Fatalf("got no frame, want frame %T", v)
	}
	v, ok := fr.(T)
	if !ok {
		t.Fatalf("got frame %T, want %T", fr, v)
	}
	return v
}

// wantFrameType reads the next frame.
// It produces an error if the frame type is not the expected value.
func (tf *testConnFramer) wantFrameType(want FrameType) {
	tf.t.Helper()
	fr := tf.readFrame()
	if fr == nil {
		tf.t.Fatalf("got no frame, want frame %v", want)
	}
	if got := fr.Header().Type; got != want {
		tf.t.Fatalf("got frame %v, want %v", got, want)
	}
}

// wantUnorderedFrames reads frames until every condition in want has been satisfied.
//
// want is a list of func(*SomeFrame) bool.
// wantUnorderedFrames will call each func with frames of the appropriate type
// until the func returns true.
// It calls t.Fatal if an unexpected frame is received (no func has that frame type,
// or all funcs with that type have returned true), or if the framer runs out of frames
// with unsatisfied funcs.
//
// Example:
//
//	// Read a SETTINGS frame, and any number of DATA frames for a stream.
//	// The SETTINGS frame may appear anywhere in the sequence.
//	// The last DATA frame must indicate the end of the stream.
//	tf.wantUnorderedFrames(
//		func(f *SettingsFrame) bool {
//			return true
//		},
//		func(f *DataFrame) bool {
//			return f.StreamEnded()
//		},
//	)
func (tf *testConnFramer) wantUnorderedFrames(want ...any) {
	tf.t.Helper()
	want = slices.Clone(want)
	seen := 0
frame:
	for seen < len(want) && !tf.t.Failed() {
		fr := tf.readFrame()
		if fr == nil {
			break
		}
		for i, f := range want {
			if f == nil {
				continue
			}
			typ := reflect.TypeOf(f)
			if typ.Kind() != reflect.Func ||
				typ.NumIn() != 1 ||
				typ.NumOut() != 1 ||
				typ.Out(0) != reflect.TypeFor[bool]() {
				tf.t.Fatalf("expected func(*SomeFrame) bool, got %T", f)
			}
			if typ.In(0) == reflect.TypeOf(fr) {
				out := reflect.ValueOf(f).Call([]reflect.Value{reflect.ValueOf(fr)})
				if out[0].Bool() {
					want[i] = nil
					seen++
				}
				continue frame
			}
		}
		tf.t.Errorf("got unexpected frame type %T", fr)
	}
	if seen < len(want) {
		for _, f := range want {
			if f == nil {
				continue
			}
			tf.t.Errorf("did not see expected frame: %v", reflect.TypeOf(f).In(0))
		}
		tf.t.Fatalf("did not see %v expected frame types", len(want)-seen)
	}
}

type wantHeader struct {
	streamID  uint32
	endStream bool
	header    http.Header
}

// wantHeaders reads a HEADERS frame and potential CONTINUATION frames,
// and asserts that they contain the expected headers.
func (tf *testConnFramer) wantHeaders(want wantHeader) {
	tf.t.Helper()

	hf := readFrame[*HeadersFrame](tf.t, tf)
	if got, want := hf.StreamID, want.streamID; got != want {
		tf.t.Fatalf("got stream ID %v, want %v", got, want)
	}
	if got, want := hf.StreamEnded(), want.endStream; got != want {
		tf.t.Fatalf("got stream ended %v, want %v", got, want)
	}

	gotHeader := make(http.Header)
	tf.dec.SetEmitFunc(func(hf hpack.HeaderField) {
		gotHeader[hf.Name] = append(gotHeader[hf.Name], hf.Value)
	})
	defer tf.dec.SetEmitFunc(nil)
	if _, err := tf.dec.Write(hf.HeaderBlockFragment()); err != nil {
		tf.t.Fatalf("decoding HEADERS frame: %v", err)
	}
	headersEnded := hf.HeadersEnded()
	for !headersEnded {
		cf := readFrame[*ContinuationFrame](tf.t, tf)
		if cf == nil {
			tf.t.Fatalf("got end of frames, want CONTINUATION")
		}
		if _, err := tf.dec.Write(cf.HeaderBlockFragment()); err != nil {
			tf.t.Fatalf("decoding CONTINUATION frame: %v", err)
		}
		headersEnded = cf.HeadersEnded()
	}
	if err := tf.dec.Close(); err != nil {
		tf.t.Fatalf("hpack decoding error: %v", err)
	}

	for k, v := range want.header {
		if !reflect.DeepEqual(v, gotHeader[k]) {
			tf.t.Fatalf("got header %q = %q; want %q", k, v, gotHeader[k])
		}
	}
}

// decodeHeader supports some older server tests.
// TODO: rewrite those tests to use newer, more convenient test APIs.
func (tf *testConnFramer) decodeHeader(headerBlock []byte) (pairs [][2]string) {
	tf.dec.SetEmitFunc(func(hf hpack.HeaderField) {
		if hf.Name == "date" {
			return
		}
		pairs = append(pairs, [2]string{hf.Name, hf.Value})
	})
	defer tf.dec.SetEmitFunc(nil)
	if _, err := tf.dec.Write(headerBlock); err != nil {
		tf.t.Fatalf("hpack decoding error: %v", err)
	}
	if err := tf.dec.Close(); err != nil {
		tf.t.Fatalf("hpack decoding error: %v", err)
	}
	return pairs
}

type wantData struct {
	streamID  uint32
	endStream bool
	size      int
	data      []byte
	multiple  bool // data may be spread across multiple DATA frames
}

// wantData reads zero or more DATA frames, and asserts that they match the expectation.
func (tf *testConnFramer) wantData(want wantData) {
	tf.t.Helper()
	gotSize := 0
	gotEndStream := false
	if want.data != nil {
		want.size = len(want.data)
	}
	var gotData []byte
	for {
		fr := tf.readFrame()
		if fr == nil {
			break
		}
		data, ok := fr.(*DataFrame)
		if !ok {
			tf.t.Fatalf("got frame %T, want DataFrame", fr)
		}
		if want.data != nil {
			gotData = append(gotData, data.Data()...)
		}
		gotSize += len(data.Data())
		if data.StreamEnded() {
			gotEndStream = true
			break
		}
		if !want.endStream && gotSize >= want.size {
			break
		}
		if !want.multiple {
			break
		}
	}
	if gotSize != want.size {
		tf.t.Fatalf("got %v bytes of DATA frames, want %v", gotSize, want.size)
	}
	if gotEndStream != want.endStream {
		tf.t.Fatalf("after %v bytes of DATA frames, got END_STREAM=%v; want %v", gotSize, gotEndStream, want.endStream)
	}
	if want.data != nil && !bytes.Equal(gotData, want.data) {
		tf.t.Fatalf("got data %q, want %q", gotData, want.data)
	}
}

func (tf *testConnFramer) wantRSTStream(streamID uint32, code ErrCode) {
	tf.t.Helper()
	fr := readFrame[*RSTStreamFrame](tf.t, tf)
	if fr.StreamID != streamID || fr.ErrCode != code {
		tf.t.Fatalf("got %v, want RST_STREAM StreamID=%v, code=%v", SummarizeFrame(fr), streamID, code)
	}
}

func (tf *testConnFramer) wantSettings(want map[SettingID]uint32) {
	fr := readFrame[*SettingsFrame](tf.t, tf)
	if fr.Header().Flags.Has(FlagSettingsAck) {
		tf.t.Errorf("got SETTINGS frame with ACK set, want no ACK")
	}
	for wantID, wantVal := range want {
		gotVal, ok := fr.Value(wantID)
		if !ok {
			tf.t.Errorf("SETTINGS: %v is not set, want %v", wantID, wantVal)
		} else if gotVal != wantVal {
			tf.t.Errorf("SETTINGS: %v is %v, want %v", wantID, gotVal, wantVal)
		}
	}
	if tf.t.Failed() {
		tf.t.Fatalf("%v", fr)
	}
}

func (tf *testConnFramer) wantSettingsAck() {
	tf.t.Helper()
	fr := readFrame[*SettingsFrame](tf.t, tf)
	if !fr.Header().Flags.Has(FlagSettingsAck) {
		tf.t.Fatal("Settings Frame didn't have ACK set")
	}
}

func (tf *testConnFramer) wantGoAway(maxStreamID uint32, code ErrCode) {
	tf.t.Helper()
	fr := readFrame[*GoAwayFrame](tf.t, tf)
	if fr.LastStreamID != maxStreamID || fr.ErrCode != code {
		tf.t.Fatalf("got %v, want GOAWAY LastStreamID=%v, code=%v", SummarizeFrame(fr), maxStreamID, code)
	}
}

func (tf *testConnFramer) wantWindowUpdate(streamID, incr uint32) {
	tf.t.Helper()
	wu := readFrame[*WindowUpdateFrame](tf.t, tf)
	if wu.FrameHeader.StreamID != streamID {
		tf.t.Fatalf("WindowUpdate StreamID = %d; want %d", wu.FrameHeader.StreamID, streamID)
	}
	if wu.Increment != incr {
		tf.t.Fatalf("WindowUpdate increment = %d; want %d", wu.Increment, incr)
	}
}

func (tf *testConnFramer) wantClosed() {
	tf.t.Helper()
	fr, err := tf.fr.ReadFrame()
	if err == nil {
		tf.t.Fatalf("got unexpected frame (want closed connection): %v", fr)
	}
	if err == os.ErrDeadlineExceeded {
		tf.t.Fatalf("connection is not closed; want it to be")
	}
}

func (tf *testConnFramer) wantIdle() {
	tf.t.Helper()
	fr, err := tf.fr.ReadFrame()
	if err == nil {
		tf.t.Fatalf("got unexpected frame (want idle connection): %v", fr)
	}
	if err != os.ErrDeadlineExceeded {
		tf.t.Fatalf("got unexpected frame error (want idle connection): %v", err)
	}
}

func (tf *testConnFramer) writeSettings(settings ...Setting) {
	tf.t.Helper()
	if err := tf.fr.WriteSettings(settings...); err != nil {
		tf.t.Fatal(err)
	}
}

func (tf *testConnFramer) writeSettingsAck() {
	tf.t.Helper()
	if err := tf.fr.WriteSettingsAck(); err != nil {
		tf.t.Fatal(err)
	}
}

func (tf *testConnFramer) writeData(streamID uint32, endStream bool, data []byte) {
	tf.t.Helper()
	if err := tf.fr.WriteData(streamID, endStream, data); err != nil {
		tf.t.Fatal(err)
	}
}

func (tf *testConnFramer) writeDataPadded(streamID uint32, endStream bool, data, pad []byte) {
	tf.t.Helper()
	if err := tf.fr.WriteDataPadded(streamID, endStream, data, pad); err != nil {
		tf.t.Fatal(err)
	}
}

func (tf *testConnFramer) writeHeaders(p HeadersFrameParam) {
	tf.t.Helper()
	if err := tf.fr.WriteHeaders(p); err != nil {
		tf.t.Fatal(err)
	}
}

// writeHeadersMode writes header frames, as modified by mode:
//
//   - noHeader: Don't write the header.
//   - oneHeader: Write a single HEADERS frame.
//   - splitHeader: Write a HEADERS frame and CONTINUATION frame.
func (tf *testConnFramer) writeHeadersMode(mode headerType, p HeadersFrameParam) {
	tf.t.Helper()
	switch mode {
	case noHeader:
	case oneHeader:
		tf.writeHeaders(p)
	case splitHeader:
		if len(p.BlockFragment) < 2 {
			panic("too small")
		}
		contData := p.BlockFragment[1:]
		contEnd := p.EndHeaders
		p.BlockFragment = p.BlockFragment[:1]
		p.EndHeaders = false
		tf.writeHeaders(p)
		tf.writeContinuation(p.StreamID, contEnd, contData)
	default:
		panic("bogus mode")
	}
}

func (tf *testConnFramer) writeContinuation(streamID uint32, endHeaders bool, headerBlockFragment []byte) {
	tf.t.Helper()
	if err := tf.fr.WriteContinuation(streamID, endHeaders, headerBlockFragment); err != nil {
		tf.t.Fatal(err)
	}
}

func (tf *testConnFramer) writePriority(id uint32, p PriorityParam) {
	if err := tf.fr.WritePriority(id, p); err != nil {
		tf.t.Fatal(err)
	}
}

func (tf *testConnFramer) writePriorityUpdate(id uint32, p string) {
	if err := tf.fr.WritePriorityUpdate(id, p); err != nil {
		tf.t.Fatal(err)
	}
}

func (tf *testConnFramer) writeRSTStream(streamID uint32, code ErrCode) {
	tf.t.Helper()
	if err := tf.fr.WriteRSTStream(streamID, code); err != nil {
		tf.t.Fatal(err)
	}
}

func (tf *testConnFramer) writePing(ack bool, data [8]byte) {
	tf.t.Helper()
	if err := tf.fr.WritePing(ack, data); err != nil {
		tf.t.Fatal(err)
	}
}

func (tf *testConnFramer) writeGoAway(maxStreamID uint32, code ErrCode, debugData []byte) {
	tf.t.Helper()
	if err := tf.fr.WriteGoAway(maxStreamID, code, debugData); err != nil {
		tf.t.Fatal(err)
	}
}

func (tf *testConnFramer) writeWindowUpdate(streamID, incr uint32) {
	tf.t.Helper()
	if err := tf.fr.WriteWindowUpdate(streamID, incr); err != nil {
		tf.t.Fatal(err)
	}
}
