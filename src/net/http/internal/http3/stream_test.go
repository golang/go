// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http3

import (
	"bytes"
	"errors"
	"io"
	"testing"
)

func TestStreamReadVarint(t *testing.T) {
	st1, st2 := newStreamPair(t)
	for _, b := range [][]byte{
		{0x00},
		{0x3f},
		{0x40, 0x00},
		{0x7f, 0xff},
		{0x80, 0x00, 0x00, 0x00},
		{0xbf, 0xff, 0xff, 0xff},
		{0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
		{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff},
		// Example cases from https://www.rfc-editor.org/rfc/rfc9000.html#section-a.1
		{0xc2, 0x19, 0x7c, 0x5e, 0xff, 0x14, 0xe8, 0x8c},
		{0x9d, 0x7f, 0x3e, 0x7d},
		{0x7b, 0xbd},
		{0x25},
		{0x40, 0x25},
	} {
		trailer := []byte{0xde, 0xad, 0xbe, 0xef}
		st1.Write(b)
		st1.Write(trailer)
		if err := st1.Flush(); err != nil {
			t.Fatal(err)
		}
		got, err := st2.readVarint()
		if err != nil {
			t.Fatalf("st.readVarint() = %v", err)
		}
		want, _ := consumeVarintInt64(b)
		if got != want {
			t.Fatalf("st.readVarint() = %v, want %v", got, want)
		}
		gotTrailer := make([]byte, len(trailer))
		if _, err := io.ReadFull(st2, gotTrailer); err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(gotTrailer, trailer) {
			t.Fatalf("after st.readVarint, read %x, want %x", gotTrailer, trailer)
		}
	}
}

func TestStreamWriteVarint(t *testing.T) {
	st1, st2 := newStreamPair(t)
	for _, v := range []int64{
		0,
		63,
		16383,
		1073741823,
		4611686018427387903,
		// Example cases from https://www.rfc-editor.org/rfc/rfc9000.html#section-a.1
		151288809941952652,
		494878333,
		15293,
		37,
	} {
		trailer := []byte{0xde, 0xad, 0xbe, 0xef}
		st1.writeVarint(v)
		st1.Write(trailer)
		if err := st1.Flush(); err != nil {
			t.Fatal(err)
		}

		want := appendVarint(nil, uint64(v))
		want = append(want, trailer...)

		got := make([]byte, len(want))
		if _, err := io.ReadFull(st2, got); err != nil {
			t.Fatal(err)
		}

		if !bytes.Equal(got, want) {
			t.Errorf("AppendVarint(nil, %v) = %x, want %x", v, got, want)
		}
	}
}

func TestStreamReadFrames(t *testing.T) {
	st1, st2 := newStreamPair(t)
	for _, frame := range []struct {
		ftype frameType
		data  []byte
	}{{
		ftype: 1,
		data:  []byte("hello"),
	}, {
		ftype: 2,
		data:  []byte{},
	}, {
		ftype: 3,
		data:  []byte("goodbye"),
	}} {
		st1.writeVarint(int64(frame.ftype))
		st1.writeVarint(int64(len(frame.data)))
		st1.Write(frame.data)
		if err := st1.Flush(); err != nil {
			t.Fatal(err)
		}

		if gotFrameType, err := st2.readFrameHeader(); err != nil || gotFrameType != frame.ftype {
			t.Fatalf("st.readFrameHeader() = %v, %v; want %v, nil", gotFrameType, err, frame.ftype)
		}
		if gotData, err := st2.readFrameData(); err != nil || !bytes.Equal(gotData, frame.data) {
			t.Fatalf("st.readFrameData() = %x, %v; want %x, nil", gotData, err, frame.data)
		}
		if err := st2.endFrame(); err != nil {
			t.Fatalf("st.endFrame() = %v; want nil", err)
		}
	}
}

func TestStreamReadFrameUnderflow(t *testing.T) {
	const size = 4
	st1, st2 := newStreamPair(t)
	st1.writeVarint(0)            // type
	st1.writeVarint(size)         // size
	st1.Write(make([]byte, size)) // data
	if err := st1.Flush(); err != nil {
		t.Fatal(err)
	}

	if _, err := st2.readFrameHeader(); err != nil {
		t.Fatalf("st.readFrameHeader() = %v", err)
	}
	if _, err := io.ReadFull(st2, make([]byte, size-1)); err != nil {
		t.Fatalf("st.Read() = %v", err)
	}
	// We have not consumed the full frame: Error.
	if err := st2.endFrame(); !errors.Is(err, errH3FrameError) {
		t.Fatalf("st.endFrame before end: %v, want errH3FrameError", err)
	}
}

func TestStreamReadFrameWithoutEnd(t *testing.T) {
	const size = 4
	st1, st2 := newStreamPair(t)
	st1.writeVarint(0)            // type
	st1.writeVarint(size)         // size
	st1.Write(make([]byte, size)) // data
	if err := st1.Flush(); err != nil {
		t.Fatal(err)
	}

	if _, err := st2.readFrameHeader(); err != nil {
		t.Fatalf("st.readFrameHeader() = %v", err)
	}
	if _, err := st2.readFrameHeader(); err == nil {
		t.Fatalf("st.readFrameHeader before st.endFrame for prior frame: success, want error")
	}
}

func TestStreamReadFrameOverflow(t *testing.T) {
	const size = 4
	st1, st2 := newStreamPair(t)
	st1.writeVarint(0)              // type
	st1.writeVarint(size)           // size
	st1.Write(make([]byte, size+1)) // data
	if err := st1.Flush(); err != nil {
		t.Fatal(err)
	}

	if _, err := st2.readFrameHeader(); err != nil {
		t.Fatalf("st.readFrameHeader() = %v", err)
	}
	if _, err := io.ReadFull(st2, make([]byte, size+1)); !errors.Is(err, errH3FrameError) {
		t.Fatalf("st.Read past end of frame: %v, want errH3FrameError", err)
	}
}

func TestStreamReadFrameHeaderPartial(t *testing.T) {
	var frame []byte
	frame = appendVarint(frame, 1000) // type
	frame = appendVarint(frame, 2000) // size

	for i := 1; i < len(frame)-1; i++ {
		st1, st2 := newStreamPair(t)
		st1.Write(frame[:i])
		if err := st1.Flush(); err != nil {
			t.Fatal(err)
		}
		st1.CloseWrite()

		if _, err := st2.readFrameHeader(); err == nil {
			t.Fatalf("%v/%v bytes of frame available: st.readFrameHeader() succeeded; want error", i, len(frame))
		}
	}
}

func TestStreamReadFrameDataPartial(t *testing.T) {
	st1, st2 := newStreamPair(t)
	st1.writeVarint(1)          // type
	st1.writeVarint(100)        // size
	st1.Write(make([]byte, 50)) // data
	st1.CloseWrite()
	if _, err := st2.readFrameHeader(); err != nil {
		t.Fatalf("st.readFrameHeader() = %v", err)
	}
	if n, err := io.ReadAll(st2); err == nil {
		t.Fatalf("io.ReadAll with partial frame = %v, nil; want error", n)
	}
}

func TestStreamReadByteFrameDataPartial(t *testing.T) {
	st1, st2 := newStreamPair(t)
	st1.writeVarint(1)   // type
	st1.writeVarint(100) // size
	st1.CloseWrite()
	if _, err := st2.readFrameHeader(); err != nil {
		t.Fatalf("st.readFrameHeader() = %v", err)
	}
	if b, err := st2.ReadByte(); err == nil {
		t.Fatalf("io.ReadAll with partial frame = %v, nil; want error", b)
	}
}

func TestStreamReadFrameDataAtEOF(t *testing.T) {
	const typ = 10
	data := []byte("hello")
	st1, st2 := newStreamPair(t)
	st1.writeVarint(typ)              // type
	st1.writeVarint(int64(len(data))) // size
	if err := st1.Flush(); err != nil {
		t.Fatal(err)
	}
	if got, err := st2.readFrameHeader(); err != nil || got != typ {
		t.Fatalf("st.readFrameHeader() = %v, %v; want %v, nil", got, err, typ)
	}

	st1.Write(data)         // data
	st1.CloseWrite() // end stream
	got := make([]byte, len(data)+1)
	if n, err := st2.Read(got); err != nil || n != len(data) || !bytes.Equal(got[:n], data) {
		t.Fatalf("st.Read() = %v, %v (data=%x); want %v, nil (data=%x)", n, err, got[:n], len(data), data)
	}
}

func TestStreamReadFrameData(t *testing.T) {
	const typ = 10
	data := []byte("hello")
	st1, st2 := newStreamPair(t)
	st1.writeVarint(typ)              // type
	st1.writeVarint(int64(len(data))) // size
	st1.Write(data)                   // data
	if err := st1.Flush(); err != nil {
		t.Fatal(err)
	}

	if got, err := st2.readFrameHeader(); err != nil || got != typ {
		t.Fatalf("st.readFrameHeader() = %v, %v; want %v, nil", got, err, typ)
	}
	if got, err := st2.readFrameData(); err != nil || !bytes.Equal(got, data) {
		t.Fatalf("st.readFrameData() = %x, %v; want %x, nil", got, err, data)
	}
}

func TestStreamReadByte(t *testing.T) {
	const stype = 1
	const want = 42
	st1, st2 := newStreamPair(t)
	st1.writeVarint(stype)  // stream type
	st1.writeVarint(1)      // size
	st1.Write([]byte{want}) // data
	if err := st1.Flush(); err != nil {
		t.Fatal(err)
	}

	if got, err := st2.readFrameHeader(); err != nil || got != stype {
		t.Fatalf("st.readFrameHeader() = %v, %v; want %v, nil", got, err, stype)
	}
	if got, err := st2.ReadByte(); err != nil || got != want {
		t.Fatalf("st.ReadByte() = %v, %v; want %v, nil", got, err, want)
	}
	if got, err := st2.ReadByte(); err == nil {
		t.Fatalf("reading past end of frame: st.ReadByte() = %v, %v; want error", got, err)
	}
}

func TestStreamDiscardFrame(t *testing.T) {
	const typ = 10
	data := []byte("hello")
	st1, st2 := newStreamPair(t)
	st1.writeVarint(typ)              // type
	st1.writeVarint(int64(len(data))) // size
	st1.Write(data)                   // data
	st1.CloseWrite()

	if got, err := st2.readFrameHeader(); err != nil || got != typ {
		t.Fatalf("st.readFrameHeader() = %v, %v; want %v, nil", got, err, typ)
	}
	if err := st2.discardFrame(); err != nil {
		t.Fatalf("st.discardFrame() = %v", err)
	}
	if b, err := io.ReadAll(st2); err != nil || len(b) > 0 {
		t.Fatalf("after discarding frame, read %x, %v; want EOF", b, err)
	}
}

func newStreamPair(t testing.TB) (s1, s2 *stream) {
	t.Helper()
	q1, q2 := newQUICStreamPair(t)
	return newStream(q1), newStream(q2)
}
