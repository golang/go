// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jpeg

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"io"
	"io/ioutil"
	"math/rand"
	"os"
	"strings"
	"testing"
)

// TestDecodeProgressive tests that decoding the baseline and progressive
// versions of the same image result in exactly the same pixel data, in YCbCr
// space for color images, and Y space for grayscale images.
func TestDecodeProgressive(t *testing.T) {
	testCases := []string{
		"../testdata/video-001",
		"../testdata/video-001.q50.410",
		"../testdata/video-001.q50.411",
		"../testdata/video-001.q50.420",
		"../testdata/video-001.q50.422",
		"../testdata/video-001.q50.440",
		"../testdata/video-001.q50.444",
		"../testdata/video-005.gray.q50",
		"../testdata/video-005.gray.q50.2x2",
		"../testdata/video-001.separate.dc.progression",
	}
	for _, tc := range testCases {
		m0, err := decodeFile(tc + ".jpeg")
		if err != nil {
			t.Errorf("%s: %v", tc+".jpeg", err)
			continue
		}
		m1, err := decodeFile(tc + ".progressive.jpeg")
		if err != nil {
			t.Errorf("%s: %v", tc+".progressive.jpeg", err)
			continue
		}
		if m0.Bounds() != m1.Bounds() {
			t.Errorf("%s: bounds differ: %v and %v", tc, m0.Bounds(), m1.Bounds())
			continue
		}
		// All of the video-*.jpeg files are 150x103.
		if m0.Bounds() != image.Rect(0, 0, 150, 103) {
			t.Errorf("%s: bad bounds: %v", tc, m0.Bounds())
			continue
		}

		switch m0 := m0.(type) {
		case *image.YCbCr:
			m1 := m1.(*image.YCbCr)
			if err := check(m0.Bounds(), m0.Y, m1.Y, m0.YStride, m1.YStride); err != nil {
				t.Errorf("%s (Y): %v", tc, err)
				continue
			}
			if err := check(m0.Bounds(), m0.Cb, m1.Cb, m0.CStride, m1.CStride); err != nil {
				t.Errorf("%s (Cb): %v", tc, err)
				continue
			}
			if err := check(m0.Bounds(), m0.Cr, m1.Cr, m0.CStride, m1.CStride); err != nil {
				t.Errorf("%s (Cr): %v", tc, err)
				continue
			}
		case *image.Gray:
			m1 := m1.(*image.Gray)
			if err := check(m0.Bounds(), m0.Pix, m1.Pix, m0.Stride, m1.Stride); err != nil {
				t.Errorf("%s: %v", tc, err)
				continue
			}
		default:
			t.Errorf("%s: unexpected image type %T", tc, m0)
			continue
		}
	}
}

func decodeFile(filename string) (image.Image, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return Decode(f)
}

type eofReader struct {
	data     []byte // deliver from Read without EOF
	dataEOF  []byte // then deliver from Read with EOF on last chunk
	lenAtEOF int
}

func (r *eofReader) Read(b []byte) (n int, err error) {
	if len(r.data) > 0 {
		n = copy(b, r.data)
		r.data = r.data[n:]
	} else {
		n = copy(b, r.dataEOF)
		r.dataEOF = r.dataEOF[n:]
		if len(r.dataEOF) == 0 {
			err = io.EOF
			if r.lenAtEOF == -1 {
				r.lenAtEOF = n
			}
		}
	}
	return
}

func TestDecodeEOF(t *testing.T) {
	// Check that if reader returns final data and EOF at same time, jpeg handles it.
	data, err := ioutil.ReadFile("../testdata/video-001.jpeg")
	if err != nil {
		t.Fatal(err)
	}

	n := len(data)
	for i := 0; i < n; {
		r := &eofReader{data[:n-i], data[n-i:], -1}
		_, err := Decode(r)
		if err != nil {
			t.Errorf("Decode with Read() = %d, EOF: %v", r.lenAtEOF, err)
		}
		if i == 0 {
			i = 1
		} else {
			i *= 2
		}
	}
}

// check checks that the two pix data are equal, within the given bounds.
func check(bounds image.Rectangle, pix0, pix1 []byte, stride0, stride1 int) error {
	if stride0 <= 0 || stride0%8 != 0 {
		return fmt.Errorf("bad stride %d", stride0)
	}
	if stride1 <= 0 || stride1%8 != 0 {
		return fmt.Errorf("bad stride %d", stride1)
	}
	// Compare the two pix data, one 8x8 block at a time.
	for y := 0; y < len(pix0)/stride0 && y < len(pix1)/stride1; y += 8 {
		for x := 0; x < stride0 && x < stride1; x += 8 {
			if x >= bounds.Max.X || y >= bounds.Max.Y {
				// We don't care if the two pix data differ if the 8x8 block is
				// entirely outside of the image's bounds. For example, this can
				// occur with a 4:2:0 chroma subsampling and a 1x1 image. Baseline
				// decoding works on the one 16x16 MCU as a whole; progressive
				// decoding's first pass works on that 16x16 MCU as a whole but
				// refinement passes only process one 8x8 block within the MCU.
				continue
			}

			for j := 0; j < 8; j++ {
				for i := 0; i < 8; i++ {
					index0 := (y+j)*stride0 + (x + i)
					index1 := (y+j)*stride1 + (x + i)
					if pix0[index0] != pix1[index1] {
						return fmt.Errorf("blocks at (%d, %d) differ:\n%sand\n%s", x, y,
							pixString(pix0, stride0, x, y),
							pixString(pix1, stride1, x, y),
						)
					}
				}
			}
		}
	}
	return nil
}

func pixString(pix []byte, stride, x, y int) string {
	s := bytes.NewBuffer(nil)
	for j := 0; j < 8; j++ {
		fmt.Fprintf(s, "\t")
		for i := 0; i < 8; i++ {
			fmt.Fprintf(s, "%02x ", pix[(y+j)*stride+(x+i)])
		}
		fmt.Fprintf(s, "\n")
	}
	return s.String()
}

func TestTruncatedSOSDataDoesntPanic(t *testing.T) {
	b, err := ioutil.ReadFile("../testdata/video-005.gray.q50.jpeg")
	if err != nil {
		t.Fatal(err)
	}
	sosMarker := []byte{0xff, 0xda}
	i := bytes.Index(b, sosMarker)
	if i < 0 {
		t.Fatal("SOS marker not found")
	}
	i += len(sosMarker)
	j := i + 10
	if j > len(b) {
		j = len(b)
	}
	for ; i < j; i++ {
		Decode(bytes.NewReader(b[:i]))
	}
}

func TestExtraneousData(t *testing.T) {
	// Encode a 1x1 red image.
	src := image.NewRGBA(image.Rect(0, 0, 1, 1))
	src.Set(0, 0, color.RGBA{0xff, 0x00, 0x00, 0xff})
	buf := new(bytes.Buffer)
	if err := Encode(buf, src, nil); err != nil {
		t.Fatalf("encode: %v", err)
	}
	enc := buf.String()
	// Sanity check that the encoded JPEG is long enough, that it ends in a
	// "\xff\xd9" EOI marker, and that it contains a "\xff\xda" SOS marker
	// somewhere in the final 64 bytes.
	if len(enc) < 64 {
		t.Fatalf("encoded JPEG is too short: %d bytes", len(enc))
	}
	if got, want := enc[len(enc)-2:], "\xff\xd9"; got != want {
		t.Fatalf("encoded JPEG ends with %q, want %q", got, want)
	}
	if s := enc[len(enc)-64:]; !strings.Contains(s, "\xff\xda") {
		t.Fatalf("encoded JPEG does not contain a SOS marker (ff da) near the end: % x", s)
	}
	// Test that adding some random junk between the SOS marker and the
	// EOI marker does not affect the decoding.
	rnd := rand.New(rand.NewSource(1))
	for i, nerr := 0, 0; i < 1000 && nerr < 10; i++ {
		buf.Reset()
		// Write all but the trailing "\xff\xd9" EOI marker.
		buf.WriteString(enc[:len(enc)-2])
		// Write some random extraneous data.
		for n := rnd.Intn(10); n > 0; n-- {
			if x := byte(rnd.Intn(256)); x != 0xff {
				buf.WriteByte(x)
			} else {
				// The JPEG format escapes a SOS 0xff data byte as "\xff\x00".
				buf.WriteString("\xff\x00")
			}
		}
		// Write the "\xff\xd9" EOI marker.
		buf.WriteString("\xff\xd9")

		// Check that we can still decode the resultant image.
		got, err := Decode(buf)
		if err != nil {
			t.Errorf("could not decode image #%d: %v", i, err)
			nerr++
			continue
		}
		if got.Bounds() != src.Bounds() {
			t.Errorf("image #%d, bounds differ: %v and %v", i, got.Bounds(), src.Bounds())
			nerr++
			continue
		}
		if averageDelta(got, src) > 2<<8 {
			t.Errorf("image #%d changed too much after a round trip", i)
			nerr++
			continue
		}
	}
}

func benchmarkDecode(b *testing.B, filename string) {
	b.StopTimer()
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		b.Fatal(err)
	}
	cfg, err := DecodeConfig(bytes.NewReader(data))
	if err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(cfg.Width * cfg.Height * 4))
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		Decode(bytes.NewReader(data))
	}
}

func BenchmarkDecodeBaseline(b *testing.B) {
	benchmarkDecode(b, "../testdata/video-001.jpeg")
}

func BenchmarkDecodeProgressive(b *testing.B) {
	benchmarkDecode(b, "../testdata/video-001.progressive.jpeg")
}
