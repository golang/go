// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jpeg

import (
	"bytes"
	"encoding/base64"
	"errors"
	"fmt"
	"image"
	"image/color"
	"io"
	"math/rand"
	"os"
	"runtime/debug"
	"strings"
	"testing"
	"time"
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
	data, err := os.ReadFile("../testdata/video-001.jpeg")
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
	s := &strings.Builder{}
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
	b, err := os.ReadFile("../testdata/video-005.gray.q50.jpeg")
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

func TestLargeImageWithShortData(t *testing.T) {
	// This input is an invalid JPEG image, based on the fuzzer-generated image
	// in issue 10413. It is only 504 bytes, and shouldn't take long for Decode
	// to return an error. The Start Of Frame marker gives the image dimensions
	// as 8192 wide and 8192 high, so even if an unreadByteStuffedByte bug
	// doesn't technically lead to an infinite loop, such a bug can still cause
	// an unreasonably long loop for such a short input.
	const input = "" +
		"\xff\xd8\xff\xe0\x00\x10\x4a\x46\x49\x46\x00\x01\x01\x00\x00\x01" +
		"\x00\x01\x00\x00\xff\xdb\x00\x43\x00\x10\x0b\x0c\x0e\x0c\x0a\x10" +
		"\x0e\x89\x0e\x12\x11\x10\x13\x18\xff\xd8\xff\xe0\x00\x10\x4a\x46" +
		"\x49\x46\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00\x43" +
		"\x00\x10\x0b\x0c\x0e\x0c\x0a\x10\x0e\x0d\x0e\x12\x11\x10\x13\x18" +
		"\x28\x1a\x18\x16\x16\x18\x31\x23\x25\x1d\x28\x3a\x33\x3d\x3c\x39" +
		"\x33\x38\x37\x40\x48\x5c\x4e\x40\x44\x57\x45\x37\x38\x50\x6d\x51" +
		"\x57\x5f\x62\x67\x68\x67\x3e\x4d\x71\x79\x70\x64\x78\x5c\x65\x67" +
		"\x63\xff\xc0\x00\x0b\x08\x20\x00\x20\x00\x01\x01\x11\x00\xff\xc4" +
		"\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00" +
		"\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\xff" +
		"\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04" +
		"\x00\x00\x01\x7d\x01\x02\x03\x00\x04\x11\x05\x12\x21\x31\x01\x06" +
		"\x13\x51\x61\x07\x22\x71\x14\x32\x81\x91\xa1\x08\x23\xd8\xff\xdd" +
		"\x42\xb1\xc1\x15\x52\xd1\xf0\x24\x33\x62\x72\x82\x09\x0a\x16\x17" +
		"\x18\x19\x1a\x25\x26\x27\x28\x29\x2a\x34\x35\x36\x37\x38\x39\x3a" +
		"\x43\x44\x45\x46\x47\x48\x49\x4a\x53\x54\x55\x56\x57\x58\x59\x5a" +
		"\x00\x63\x64\x65\x66\x67\x68\x69\x6a\x73\x74\x75\x76\x77\x78\x79" +
		"\x7a\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98" +
		"\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6" +
		"\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xff\xd8\xff\xe0\x00\x10" +
		"\x4a\x46\x49\x46\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb" +
		"\x00\x43\x00\x10\x0b\x0c\x0e\x0c\x0a\x10\x0e\x0d\x0e\x12\x11\x10" +
		"\x13\x18\x28\x1a\x18\x16\x16\x18\x31\x23\x25\x1d\xc8\xc9\xca\xd2" +
		"\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8" +
		"\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x08" +
		"\x01\x01\x00\x00\x3f\x00\xb9\xeb\x50\xb0\xdb\xc8\xa8\xe4\x63\x80" +
		"\xdd\x31\xd6\x9d\xbb\xf2\xc5\x42\x1f\x6c\x6f\xf4\x34\xdd\x3c\xfc" +
		"\xac\xe7\x3d\x80\xa9\xcc\x87\x34\xb3\x37\xfa\x2b\x9f\x6a\xad\x63" +
		"\x20\x36\x9f\x78\x64\x75\xe6\xab\x7d\xb2\xde\x29\x70\xd3\x20\x27" +
		"\xde\xaf\xa4\xf0\xca\x9f\x24\xa8\xdf\x46\xa8\x24\x84\x96\xe3\x77" +
		"\xf9\x2e\xe0\x0a\x62\x7f\xdf\xd9"

	timer := time.AfterFunc(30*time.Second, func() {
		debug.SetTraceback("all")
		panic("TestLargeImageWithShortData stuck in Decode")
	})
	defer timer.Stop()

	_, err := Decode(strings.NewReader(input))
	if err == nil {
		t.Fatalf("got nil error, want non-nil")
	}
}

func TestPaddedRSTMarker(t *testing.T) {
	// This test image comes from golang.org/issue/28717
	const base64EncodedImage = `
/9j/4AAhQVZJMQABAQEAeAB4AAAAAAAAAAAAAAAAAAAAAAAAAP/bAEMABAIDAwMCBAMDAwQEBAQGCgYG
BQUGDAgJBwoODA8PDgwODxASFxMQERURDQ4UGhQVFxgZGhkPExweHBkeFxkZGP/bAEMBBAQEBgUGCwYG
CxgQDhAYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGP/EAaIA
AAEFAQEBAQEBAAAAAAAAAAABAgMEBQYHCAkKCxAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1Fh
ByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNk
ZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT
1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6AQADAQEBAQEBAQEBAAAAAAAAAQIDBAUGBwgJCgsRAAIB
AgQEAwQHBQQEAAECdwABAgMRBAUhMQYSQVEHYXETIjKBCBRCkaGxwQkjM1LwFWJy0QoWJDThJfEXGBka
JicoKSo1Njc4OTpDREVGR0hJSlNUVVZXWFlaY2RlZmdoaWpzdHV2d3h5eoKDhIWGh4iJipKTlJWWl5iZ
mqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uLj5OXm5+jp6vLz9PX29/j5+v/dAAQA
Cv/gAAQAAP/AABEIALABQAMBIQACEQEDEQH/2gAMAwEAAhEDEQA/APnCFTk5BPPGKliAB718W7H2j3Ip
VUuwJxzTfKXacde9VBhYRUBAyO3pTmUAbSMU5WGmybywzHGAMdelPVFC+n1qXZCuyaJADxjj2qzbBMAP
xz1rKaVib6ltLcFvlIx2pLy0dwuAMMBnH1rFON9RNsszAZPFEYHldPzrOy3KewmBk9qUABugxjtTVmiW
xWRcjp+VJtHXgVL3K6AgBDdM9eRTNzAZViOe1VyxaJavuf/Q8aW4mUcSGpo764AyHz+FfnnJBvVH1UsN
CS1Q/wDte4Trip49ecA7g3FSqMW9zlqZandxJ4/EKADcSPqKni8QQMT865qOSUNjiqZdNbFiHWYXz84N
WE1KNsfMKj2zirHHPDSj0JFvo2H36d9pUjg1sqykYOm0KbgY60omXPXmr9pFkco3zBnrQzjGcnrRzp9S
bEbuOvao3fisZSXUpIYWGKGcbetTCSswsxnmACkYrtNSpJ2YNM//0fnK1BD7sDg9KmUHeOe/Svid3qfb
SdmQ3AHmnr1pGBC5z19a0hohNiJkensM1J0yCKmY0yZR82e+BT1BxnpmpepN9SRCR0NSpweOoPWs6isr
ijuWIZGBA/lVwzMVFY8ibuhXEfr+tOz8hIqUymhRnJGTSc5wBVRRDFPXBHJpB3qdmV0EX7vXmoyfl685
p2dxWR//0vFsHZ9TQv3T618Bqz7PSwwn1phPXpSWrQEUhIx0NVXc7j0rSNwViCS4dWYpJj3BpBqVzGy7
ZmHSq9kpblSpxkveQ+PX7uMf6wEDtU0fi24TAkX8jTeCjJaaHDUwFN7aFq28aL/GCMGrtr4xtHGGkA+t
YTy+a+E82eAa2LsXiWzI5mXPHercOsW8hwJB+dcUqVSCOKVFxdmiwl7E2MOPzp4nQ9GH51jzNbmUoOIC
TI4okOaUXoybDCevNBPHX8qIO4mf/9P52i4dix5zjp/n1qZFBCmviL6an2kt9CGcYnJznJpOBwegq4vQ
L9xIUytSkfu/bv70p7j6EnQgjHSpFGVqXclkkaHb1+makUHgdazm7IFuSKOasrnjis2+oDm9qnIHlgd/
es7gxqjkt1NLwH4xTTEhjkhutM3D15oGhkcnBGRTDKu3A7H1rS3cLn//1PEhJ8uM557UvmDaa/P7a3Ps
xpZcZ6mo5WG45pdUC2K8ko5JIzWfcTqu7HPHrW9OLbKWhSluVLNz3wKrS3I3KfcV1Rg9CrpXK8l0F7io
pLnLnJHGOldMYJGMpu5XNwuxjyRTBcAAjd1HWtfZmPORy3WAWWQDOM4PWtHRru6DFlmY88ZqKsVyXaKp
QjOoa7axe28G/cWqhoHjO/n1WeJwSkS9c981wUcFCopPayFj8JC8VFbs6e38VldvmHHFaMHimAoCzDB6
V5U8FKz5TlrZU4/CXYtdtnXIarEepW7jAcfnXGqEoKx5tfBzh0P/1fnqEAsc/wB6pI9owAD1618Qn3Ps
35EE4UzHrx79aXaMcdaqAMWIADvj271IMeXg59KUmNLQkUDfjb1FSLxzg0pWJRLGAQAeMVIoA+uaxlaw
0SoF/u1KowwwDUcwuo9wMjrUrY2ZPOKy0KY1T1NMdwG/CtBEFzMqnIPNUZ75FBJP5mtIQvYfoU21JFVs
N271AurRE/e611xw73Yj/9b50GsQhOXHWnpq8JX7w4PWvjPq76H2fzHjVYCud9Q3GrRAZDUvq75kNbMz
7vV0zjdjNZ82pqzMcj7tdlPDtIiVWKKct+AxwRxUbXi7VPJAIrZUdEZOsrsga8DFgelQtd98g5P6V0Qp
GE6qIUut2cZ470kd2FjYc4Oce1bSpJ3Rzxq21GNcDZhSeg710ujKRbKzAg5rkxceWnqd+XtOo7bD9cl8
qxLDPHasXwUvmyXU7Lgl8cegrnw2lGbZ14l3rU0bl3gMQCRgVU1y7WytUZQzMRwBXPRhzWRvVny3ZW8N
6xPdXBikiZc5IOa6GG6nDsd5xnAyfaliqEacrGOHarx5pI//1/nuL754HWngEkYx1r4VWsfaMjk4mP8A
OgnjPH1rRMLKwR4A2jH1FPA+TNRIa0ROvQcY4p4GF/pUskmi6+gqRACvPrWMnpca3JABjFSKCQOnFS2u
o7E3XBOKcR8ucdKzUkDGSHGemKpXchVuP0rSmDMfUrl1J5rn9TvnVCc9OtelhoJtDekW0Yb6pId3zdRw
RVT+0pAPvc57CvbhQVrHlTxD3P8A/9D4tbUpTH1I7cU/7ZdMnyqcdsV5vsErXPbWJbHLdXzYQDBY8c02
6udQjyGVuD1FHsqfMridepZ6ED3s4IDqeD0I68VEt7J5hy3GO9aKkuhPt2BumeRjnv3pJLlgwBYE8ZqH
T2GqujYLcuWYbhj0zTHm5B/vcGtVBLYzdRtEcUueoGB3FOjmBjcBhx2NNx1IhO+uwtqd93EgA5YcV32n
IqwrnAz2rzMx+FI9nKldy+RmeMpfLs8DGTxTvBUKw6Csjry2WPbrXHB2wzfdnbUu8SvJF4xh1LDAJNU9
UtVmDs3IiGB6CuenNx1R0yjfRGd4aRTqJdFG1ARXRgANg4/yK0xbvJehnhlaL9T/AP/R+e4x8xx609F+
YZ718L6n2ju2RzqTKcYpQMjsc1pHTQWlgjUjGVH0qbkr0BqJSKRMi+uBx3p8a5HYVD8yb32JY15FSKpx
nisp6RuNbj1BzUyrnkmo6FEqrz7U/advHOazvcRHIuSazNXDpbSSJjeqErnpmrp6CueeXusahO5zKi8c
7VrPuGklUiSQtkd6+po0YQs0edVrTd1cqeSoJOB0xUCxpnouAecV33e558rbH//S+KdmFHTk1btywUqc
YNcEnfc9SGl7DyAJ1AHIParx+YZ4/Guea2OmC1dhGjQn5kXNQtbQFiWiTlfSoTa2G0nuU5bG2aQ7V2jP
JU+1RSaXC2GjuApyOorX2slYz9lF3sV/7MmViFljaoJLG6SQbkyDXQqiZzyg1Yg8i4jBJjIBpsaPyXXB
Psea1TTMJJqysaeh2u/UUfP3QCBXdQJtTpivFzKV3FH0WURtCT8zmPHcrhkRSOWro7O28rRYIgOwGB3r
mnph4+bOxNvEy8kWFi+ULxwKzNRkMemSPj/WMT+FckNWdLv0KPhCMmGSZl6k1ulC3zY5x2+la4r+IyKH
wH//0/nuIHB9c9KevUAHk18La60PtHuRy/64+lOGBniqXcOlhUIxwB+NSrynSpndFImQc4A7d6lQccdR
WcyUyWMccDnPSpVAwM1nLYaeo5BjrUyjFTugJIwd2Kfgkc59qyTs7Axkigqao3qBkYdiMVpTeugHmF7b
hbhl2/dJB/OofJBTHfp9K+ppTbimeZOK5issYG5W7VWdBnHXB65rrjJs5JLof//U+LYtu7leM+lSpxIR
7VwO90eskrNkqZLo3PXHoausxI4wa557o2p6JitnOCoqvI3zEkdF5qIrUuW2ogO1iWHeibazIQncHA+l
DT0aCyaaZGNm8kA9PSl2qy9SB78Veq1ZCs9BkOGUrj86zdQGbllVMAe1bQdpGE1eBo+FoCbgtxkY966+
E4hOeo5rycxleR72VwcaRx+t/wCmeJ7WAdDIOPpzXbSpt/dkcRr+tZ4j3aVNLzNKOteo/QjuiY7Jm6Ej
ANYnitvL05YRxwOf8/WuXDK8l6nVUlaLZb8NQeXpijgZB/M1oIpyPzx74pV2nUbHTVoJH//V+fYhgnnv
Txy4GBXwse59m7kMyEzkj8qkQfKatdgewIo7nIqdQAnXms52RSehMoHHPapY1wMAgVnKzFtqSwjjg4qR
VJHXIzWc2rDiPVeeD+FSDqKh2sBKo54p+Pl61jbQG9RrDIPNU7teT6VpCztYDzfxGskWtXESdN5Pp15q
gN2GZpB0r6ig17OL8jz535miCSPdnaxHHpVV48D7xIB7iu2LOOS7H//W+MCoeIDcc5p4VhIMDkDvXnpq
+p6zu0SwZZlVm6HJFWyRg89MdawmkmrG1NtpiMcY5OevFQ7AWOT0FSkkU9UKUPmEh8jt+VMdGLDLYAIz
xUtrQfLo7Mj2SHjePrSspxgEk1rdGSTGJjymLEZArOjAd5GLHk9DW0NGznqa8qOj8IRHBbrnnmugu08u
1ZiSMn868LGz/eH1GAVqKOW8LR/bfG4c8rCCx46HpXZspk88jHzMf04pY7eEfIjDO7nLz/yKmqh/sjwR
EFwAemcVhamkmpTRxKpyCN2RWeFsveZ01FpbubsEaRWyqhAxnH5YpxIx8rf/AFuK5W3Jts2Wisf/1/n5
SSxHOM+lP7jGa+EVz7R2IpATN1IIpwB55NaJ2FuhYzx3PvU69OQaio7sEiZOvfpU0YwmMVnJ26DRJH2G
DipUyR361jN6FIeq8/0qUdBxWbkCRIg/D6U8j5e9Rza3ExrA8nmqt0Dkmri9BnnfjlSmvuwGQyhulYkr
yL86DANfTYRp0o3PPr3UnYbBOWU4zz7VHIGIJVjkGu1x5Tl5ro//0Pi3fhgMHJPXFWCeQwLe9ec+jPXj
1JIM7gw44qy+WPUjkcVjPdGsNmgdsNkjJFQ7mMhAB5FRHuXJ6WRIw+VwCc9KbtPy5JyCKgdmxhBDNj8s
U1Cyr0J/rWultDOzTuMuSiWjlT97r7HFZ1nkk4bIPXiuiD3uc00rqzOy8Lw+XBuJPQcGrXiGYJYMwJHB
xXz1d89U+sw8eWmkuxi/DGP/AEm+vHycYUE/jXXu6w2vzdcfiaMw1qpLsjnwi/dt+bKCn5nlw2W5Gacw
GD2wB+dcq2O4AnyADoM80QLukUsp9f0qb6XHuf/R+fkOWNSfwjnivhUz7Nq2hFJ/retPA4PWrWuwul2L
FjAA6VMMFeTms5PUpbEw6/hxUyfd4as5PUETRds9KlA+Xk96ym9FcaHJgGpOv1rPUpIkXg8mnNnGaz5r
aCaEPeqtx1OT0rSL2sC1OG+ISquoQuT1UjP4/wD16wGEZUYGPevo8E26UThrpc7G7ICDzg+1ROmF+91O
K7VKVrHNyxWx/9L4uuVAcH371JvKqScFPU1597pHqtWbZNZnc+QxI9atv8p4z9ayqPVI1pr3WyPBLDGf
qKYnExyeQKlFPQXH7zgdetSk8rgEnis29i0lqxijLEjjt1poU7iVHHpVX3uRbsZl+2IvLX+I56U/TUUA
KxGSfSulu0XY5oq80drpcZSzHvjpWd47fy9O2g8kjpXz0ZJ1kvM+skrUnbsSfDm1C+HlfJ/euXIPRq3b
lleQRYBCg5HrSxk+au/IxwkbUokRw0u0cBFyR70wEEbm6sc/gK5YuyZ1PzFVgVG4ZIzmpbTaJMt07+3F
Q9i7n//T+foic55yTipRkYBBxXwaPtXuRyZEg4pWII4qk7C6BFwAf51MhG31+lZ1Frca2LKHn8qlDY6L
UNgl1Jo+2akQ9BWVR9xpDgffrUq+wrO7tdDsSIPUUpPHvUK1xMM8HA61WmwWOB+NXENjiPiMhE1sw9WH
8q5vqRnjivosD/BXzOKv8QkKgZBA6ZpV27MkDOa7ObsYI//U+MdVGxlK9zninqd1sCQM45rzYaxR68vj
YtgT5h6jvV6Q5X0+lZ1n7yLofCxhOenfFMTI3cdRzWV9DWw5ARI3qPSnMSqKCOSRUy6FRurjFLjPp9KA
xx06n1q1qjO70Me6YtcOOcKcH1q9oqF5l75Oea6KtowOagnKol5neWcSJaocdgRzXGfEm53ERKfvEV89
gvfxCPqcXLlw8/Q6fwkph0aCEg4VB/KrsDbmcnA4PWoxFnVkxUVaml5IgR9sMj4+ZzSTuTjcOB0/CsUt
zo0VrhCQiF2GcAn/AD+dWLRlYZPQ8cVEk7aF+p//1fAIuvfOakxnr+NfBJ2SsfaN6jJRiUA9RSheCMfn
VXEtgjUk/wBTVgfdBwOfSs5stbE6g7unYVLGpwAazYvUmjHHanqDx061lLazKuh6DHBFSID27VEthkin
5cUHOPxqLvqJiEYziq8/FaQ8hHH/ABEVvIhYYyHNcsrSZG5RyOtfQYC3slfzOPEX5tAA+amHIjO31ruu
rHNa7P8A/9b411QMIwSDnNR2xYQNkjnnkV5sLctj15JqRLZjBzweeSKuycHkD8qyq6tF0rqLI2OTnK5p
sGWbBHQd6zWxo3qSdXLYxTpPvLnvjrWdr2LvuNYYUnj6Uxyu4/KMrVx6ky6GOGLSOwXIYmr9n58UQeFg
svbcCRXTVty2ZzYZOVRcpvDW721tv9LsBIpAHmQNn9K4zxPqSX2sK6hljDDO6uHBYWKre0i9PxPTzDFS
VDkmtX9x2mm65YG0REnTccAc1rx3EJgbbIpyvUGuDE0JxleSO+hXhUj7rGK/7uNcj1P6UmSU+fHPJrlS
5bnXe9mA/wBSQDxzkfh/+qp7YbIipbaOufwrN7WLR//X8DTO7I9e9SJnAIHNfBPY+ze4yQEPnGacoHof
amvIELCDnnqanXoBzUSaRSRMo5J744qRBxzUSelwRPEMingdsVk7pFLckA5p44OTUXuh2HqOKRskE4qV
sKQADkYqvcj8aqD0shWOV8fqDYqc4w4rj5D365Fe7lz/AHdjlxC11CNhyRnp0pqkYOc9a9G/c5bLof/Q
+OdUH7sDnOeKrREgEN6V5tN+6exUT5rlm1IC9ec9qssBg5xzjNYzdpJmkNtRhJzgflSRDqD6flU7FkqZ
55+holblfUVk90aLYYxJH3agvGCQPkjJ6DFaw7GMtrmdYpkcg1q6fsEwB6Ct8Q73M8DpKLZruu+3ZM43
ADNctfeHt120cEjO3U7scmuDCVvZNux7GLw31iNmypLoN5byByrYzzxVfzdTtXcxzSAD3r0oYinW0Z41
TB1cPrBlyy8S6nF9+QOMba1LbxepwsyFcYztrnrZdCabgb0MzlCyqGrZ+IbK4U7bgDg8E4rXtL+CQBfM
VgTn9BXi1sJOno0e5SxVOqrxZ//R8DiGScip0HOAOa+CfkfaDJV+fpRjBIxT6iHRjnIHFTAYXBIqZ7lK
1rEygZxUsYI68Gsm7gkSQ59PwqZRxx681lK/LYaHpjI7U8ge1ZK3Qdxw5XpijAOelGoNCDPOOaguOf8A
CnFCOa8cJu0uTI6EHH41xLou3gYBr3culaLXmc2JV7DIx8pXHFNAxknGc16b0OM//9L44v8A/V5wMiqy
/dbI6j1rzKb0PYq7li2C4HHHrVmT5Rx9ayqN8yLppKIwEj6+lPhAGfzNQ9rmqexKvcAYxTTjeM88isty
xhI3mqWrNwqADB5ranozCo9GQaeBjByR1q2GIzjj2rasryMaErRVieK/kjwHbIGOBVrSbtZNRmlfAB+U
ZP0/wrinQsnKPY9eljVJqM9zYgMTMAjoRnoRUV3ptrMrl4UY+o4rzIzcHc9Nx5kZc/hu1lUlBszWPqHh
Z1lYxHJHQZr0cPmElpI87E5dCptoZl1o9zADuVhwfaqqz39tIPLeVMDoPpXqUqsKyPHr0KuGfun/0/Bo
Rzlh+lToOBXwGltD7RjJQcjtSqODn9RVLyEEY96nVflzis6l3YpaEigdscCpl5XqeKiaVgJIhn/61Sgc
AdOaylqkyluPXv7U8gZx/Koih9R+ML1ppHynIoRLEAqKYZ6njFVHRXBnP+Lk3aXN0yFz+tcM6j15r2cv
ejOfEdLjMDb94ZHaoyvy9O/Y16mqOQ//1PjfUcBcnjJ9KqggRldo579682krxPWqO0i1AM7c59OlWJMA
/lWVTdGtJaNiLg89afCMkjOOO1ZtGu5IpCs+emf6Uxug6Y45rO3UruiFmKydvrVK8ZZLogAn+ldNOPU5
pvox1pH+7JzhumKlVfkOckmnNq5MI2SGScE5HTBqWDcBkMvJ6CpuraFWd+w3ULqW3gMsb4fOOO1VNP8A
FGoxsyy7JFA6kYNKGDp1Y+8aSzCrh5e7qjesPE9s8eJh5eRnIrShvbG7AMUqZ+teVVwU6TbWqPaw+OpV
0lezJWgV4yAVYc8GqV5pFpMMyW+O+V+lc1KtKD906q1KMlaR/9XwqPg9e9SqOQQetfnyfc+0Y2UfOKFy
B25q1K+grCx564FTL93rilMaRKvXtz7U9OnfrWbtbQdyaLoOtSr/AFrOSstCl2HLnPNP5z1qNEA9s4FM
7daSVhMAcZzTJlOD1prYDC8VoTpdzjP+rbH5V5w/nDkj9a9rK2mnc5sVeysQxmYMd4wDUkLM3LZFeu0u
hxan/9b44v8AG35cnnNVArKM+tefS+HU9WoveuizCQCBuOT61ZcnGcgmsZrVG1N3TGrk4x3qWBQGPUHF
RJspa7kjAFmwOp6Y9qSXPyDtkVgnsa9yFxhWYrj2rJUtJcFgB75rqpSvc5K0bWSRetV2Rg98c8d6UZCs
M8AZPNTLWTuVFWiiIgjcBnDY5NToCsZAyDnqKUrtJDitWUfETbYI1Ixz7ZNYkXyzMAMevHSuvDfCceK3
NKGIG0Ppj73pVdi0F2MMQDj7pog7uSYTXKotGxYajeQwA/aHOAeG5zV6HxJLCdk8KuvTK8dvSvNq4OFV
6aM9ajmM6SSlqj//1/DIOpHGaljG4cYH4V+e62PtWEi45pvzcjIwKuLsS9RYwM9s1Jj5Md88molqUnYl
Xlue1SryKlpodiWLGB6GpBWU9FcEPQd6eBgjNRYpCn7opDyM0Ru9RMQdxxzSODVR12F6mTryFrWVcfeU
j9K82ctvxxXrZY7XRjiF7qEjUHgkYFQuuCeB19K9dO5xSSsf/9D44uwSmM556VXkXMJHGenFeZTvoexN
Ilg4bBOTjrip5uQcY4x2qJ3ckVDSLQ6MdCPpT4uGIHGOaze1jRJbkjFgDgYpjEgqxI7dqyWhpuQ3rkQS
HjpWfaR9/wCf1rqp6RZy1Peki8MGMfNx7d6jc7Yyc4ye1JptjvorjBneDw2QMVPEHBIOFwR3qbXF5ox/
EDl71U4ODiqDrjJABIrsoaRSRx19ZNmrbD/RgScAjsag1CF9pcBTxWMXaTNZRbirDrF3CbSQTg9f5U+Z
QJxwFBA/lRZc1yrvlsf/0fD4fvE1MnYk/nX5/wBD7R7iP1yMdaQLzRF9LCa7hGvOf0qXHy9qUtxkgXnt
UqDk0pK7sCfYkUcYA6VKnIFYzWha7jhT1Hes42TsC1FIyOlNYEU4oT8hB3pGzt68U1tYVzP1IZQ56Yrz
e6UC5lUrwGOB+Nepl3xNGddXiV0Y8jGPw6U1slSCec9a9m6OCx//0vji/wAiMY45qux3Rbh/KvNprRHr
1N2iW3z5gHY81ZnA3AnipqfErF01aDuPhwXAPHSpME7jyOK53pKxt9kT5vmB4IPHFMkwcEnkkVPkPpqU
75zt8sDBJ+tNtVCt8x59q6rWi7HKneSuTxAhCAQOcc0yQEA4br2FRfVqxfK7LUYw4weCu3n1qbdiNmOM
EgY9KV9CXo2YF+VMyNx8zE5qMgsrhccdq7obK5xT3aRdtTi2VmPGMdc+lWJgr5XAGMfrWElq7G8XsmUx
C8ZJjOQQcgdakVXZxv4PbjvijmTJlBr7z//T8RjIGPc1KhXqPXivzzdH2bWoSD5sg4INJwRzz9aEw2BM
dM5qTjb60SdykSA88enepEOOKU31EkTREdqkH54NZT1Ra0Hd/SnHip0Q/IVjwPWmsePehPqiWC9T1pHH
B5oTaFYoaiOOn415tqm4X842jiRh+tenlvxMzrfCV05JLDHTgD3ppYFCB6/Svb1aOHbc/wD/1PjnUv8A
Vct3qq53IMeleZDZM9ipu0SRYDAkkewqxI24+w6VM90EL2aJYCN5AbgY/lU0R+9z+FZS0N076DZSNzc5
yc0xyoUfh25qG9mU7XZnXLZuTgnaKsW4zjqRnriuiStHQ5Y/GOxgEVFuwpBA9+MVKsytUOjILcZ7daNQ
ZEt3O7k8fjU3ldDbVmc/cPm4QYA/rS8gMD2zz6139EcF7tli2lVIAhVsYPap45lZiFG08DnrWLT1Ztde
6kidMGMZHGD3pkqjcOcZ447cVjF6o0nZxbP/1fElB7HOD1qVQOOtfnSVj7VsRwNx60Ywp4PFOCVwvoIg
HB9e1SAAjGcUpa7AiQDkf0qRRxgZGaUrWuwJYwOByalXGB6A1nOzRS3FAI70/GSOtTtuMVunemHGKUe1
iRRgk0hwKqPkDZS1DkYFed+JEEWs3Q5GHz9M816GX6VPkZ1V7pQibOck8jrUbgDIA4r3epwJ6an/1vjf
U9vl43YyelUwR5RGTxx0rzaauj1qrSlckiA3D5j071aYjccnHSlJXaHDZsntzkY3Z96lQjnnt9K55I3T
GsRyDnI9ajfKKOemDxio7Iq+7M3JeVieSecYq7AMKF3Fj6V0zVlaxzQfvXGDpy2McdajjydwIO2h2Hd6
ND41G8nAGMdD1qLV2zAFORl+R1qEtVcTdkzBl5vSCMgNwKlVG3MFJ4967r6I4bO7JY1BjJJ7Y6dDVqyQ
CYkqTjvWEn7rN1unYtYQLntg9+tV2+aQBlbPTHrxWNPzNqux/9fxWMDP41IoUYr87WzPs3uEgBJ9uKAB
tINCSbDoJGAMcdKkAG04/KiSXQpPQkXGcY7VINp56VLBbksYBxUgAx0rOadrDT7C45pxA4wKVkUhX7ZF
NODzRG1iGAA5NIwGeB1FNLuBTvEBI7V5/wCNFP8Awkl1tIABUdP9kV35cl7VehFbWBlxbNuD8x9aY20Y
xjOa96xwWstD/9D451XAhBOOTxVRlURdOa8ymrI9epu2SW23cMgDHb1qw4BcrgY9KJ25kOn8LLFsArYG
McVIu3GdvOOTWLV2axeliOYgM+FzmoL1isAwME4GKVloDvqVoI9zEEAMMGrbIAmFUdcZrSbS0MqcWQBg
0WMg4PalhCFCdpHWiXUFZtaDtqhywBwMY9Ko6sR5oXPTn2p09WianVIyEYebwMEnPNTL5eXBA4Hautp6
WONSWt0TQMhUgjKj1q5bhQWKMDwBxWE20mbws3ElXy9pUEDqMd6hfaHUgE7B2HtWUO7NKiVrI//R8Vj4
br37VMuT9TX55HY+ze+gMM5PSkzwR1AoWu4IRRg9akPC8HpSlfRFLXYeCN3TkipV60ulmBJGeR71IvTP
Ws5K6Ghwp3FQkMH6elNJ461didOoq9SaRjxSjqwuVbgZkH1rzrxWwk8QXj9hIR+XFejl1nVt5EV/gM2I
qCTjPTH50x+eRxya961tzzt9j//S+O9WJ+zjgDmqJYGDJYZNedSSaPXqu0tew+L7wPTvmrBO5x7Y5qZf
EnYUPhZdhUZyO3pTgcOxGOR0rnbu7M6fhK9yVEvPy59O9QX75VBjnIoS+ETa95DLbdu6nnp7VYGSMk59
vxrWemxlTZCvMe4BTg/nSRKHVsjGfpSe90C1VmTsG2H2x3rF1J1eVxgcHr+FVQWtyK7srFGAgOrHqDnp
7VPGNzO/O3kkCut7pnGno0idCCNyt6Crca5ZiDjpgVzXsnc6X71rDyBgcbtucn8KhjBMpOfrz7VlTdtT
Sqm7JH//2Q==
`

	data, err := base64.StdEncoding.DecodeString(base64EncodedImage)
	if err != nil {
		t.Fatalf("base64 DecodeString: %v", err)
	}
	if _, err = Decode(bytes.NewReader(data)); err != nil {
		t.Fatalf("Decode: %v", err)
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

func TestIssue56724(t *testing.T) {
	b, err := os.ReadFile("../testdata/video-001.jpeg")
	if err != nil {
		t.Fatal(err)
	}

	b = b[:24] // truncate image data

	_, err = Decode(bytes.NewReader(b))
	if !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Errorf("got: %v, want: %v", err, io.ErrUnexpectedEOF)
	}
}

func TestBadRestartMarker(t *testing.T) {
	b, err := os.ReadFile("../testdata/video-001.restart2.jpeg")
	if err != nil {
		t.Fatal(err)
	} else if len(b) != 4855 {
		t.Fatal("test image had unexpected length")
	} else if (b[2816] != 0xff) || (b[2817] != 0xd1) {
		t.Fatal("test image did not have FF D1 restart marker at expected offset")
	}
	prefix, suffix := b[:2816], b[2816:]

	testCases := []string{
		"PASS:",
		"PASS:\x00",
		"PASS:\x61",
		"PASS:\x61\x62\x63\xff\x00\x64",
		"PASS:\xff",
		"PASS:\xff\x00",
		"PASS:\xff\xff\xff\x00\xff\x00\x00\xff\xff\xff",

		"FAIL:\xff\x03",
		"FAIL:\xff\xd5",
		"FAIL:\xff\xff\xd5",
	}

	for _, tc := range testCases {
		want := tc[:5] == "PASS:"
		infix := tc[5:]

		data := []byte(nil)
		data = append(data, prefix...)
		data = append(data, infix...)
		data = append(data, suffix...)
		_, err := Decode(bytes.NewReader(data))
		got := err == nil

		if got != want {
			t.Errorf("%q: got %v, want %v", tc, got, want)
		}
	}
}

func benchmarkDecode(b *testing.B, filename string) {
	data, err := os.ReadFile(filename)
	if err != nil {
		b.Fatal(err)
	}
	cfg, err := DecodeConfig(bytes.NewReader(data))
	if err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(cfg.Width * cfg.Height * 4))
	b.ReportAllocs()
	b.ResetTimer()
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
