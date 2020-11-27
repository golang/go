// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lzw

import (
	"bytes"
	"fmt"
	"internal/testenv"
	"io"
	"math"
	"os"
	"runtime"
	"testing"
)

var filenames = []string{
	"../testdata/gettysburg.txt",
	"../testdata/e.txt",
	"../testdata/pi.txt",
}

// testFile tests that compressing and then decompressing the given file with
// the given options yields equivalent bytes to the original file.
func testFile(t *testing.T, fn string, order Order, litWidth int) {
	// Read the file, as golden output.
	golden, err := os.Open(fn)
	if err != nil {
		t.Errorf("%s (order=%d litWidth=%d): %v", fn, order, litWidth, err)
		return
	}
	defer golden.Close()

	// Read the file again, and push it through a pipe that compresses at the write end, and decompresses at the read end.
	raw, err := os.Open(fn)
	if err != nil {
		t.Errorf("%s (order=%d litWidth=%d): %v", fn, order, litWidth, err)
		return
	}

	piper, pipew := io.Pipe()
	defer piper.Close()
	go func() {
		defer raw.Close()
		defer pipew.Close()
		lzww := NewWriter(pipew, order, litWidth)
		defer lzww.Close()
		var b [4096]byte
		for {
			n, err0 := raw.Read(b[:])
			if err0 != nil && err0 != io.EOF {
				t.Errorf("%s (order=%d litWidth=%d): %v", fn, order, litWidth, err0)
				return
			}
			_, err1 := lzww.Write(b[:n])
			if err1 != nil {
				t.Errorf("%s (order=%d litWidth=%d): %v", fn, order, litWidth, err1)
				return
			}
			if err0 == io.EOF {
				break
			}
		}
	}()
	lzwr := NewReader(piper, order, litWidth)
	defer lzwr.Close()

	// Compare the two.
	b0, err0 := io.ReadAll(golden)
	b1, err1 := io.ReadAll(lzwr)
	if err0 != nil {
		t.Errorf("%s (order=%d litWidth=%d): %v", fn, order, litWidth, err0)
		return
	}
	if err1 != nil {
		t.Errorf("%s (order=%d litWidth=%d): %v", fn, order, litWidth, err1)
		return
	}
	if len(b1) != len(b0) {
		t.Errorf("%s (order=%d litWidth=%d): length mismatch %d != %d", fn, order, litWidth, len(b1), len(b0))
		return
	}
	for i := 0; i < len(b0); i++ {
		if b1[i] != b0[i] {
			t.Errorf("%s (order=%d litWidth=%d): mismatch at %d, 0x%02x != 0x%02x\n", fn, order, litWidth, i, b1[i], b0[i])
			return
		}
	}
}

func TestWriter(t *testing.T) {
	for _, filename := range filenames {
		for _, order := range [...]Order{LSB, MSB} {
			// The test data "2.71828 etcetera" is ASCII text requiring at least 6 bits.
			for litWidth := 6; litWidth <= 8; litWidth++ {
				if filename == "../testdata/gettysburg.txt" && litWidth == 6 {
					continue
				}
				testFile(t, filename, order, litWidth)
			}
		}
		if testing.Short() && testenv.Builder() == "" {
			break
		}
	}
}

func TestWriterReset(t *testing.T) {
	for _, order := range [...]Order{LSB, MSB} {
		t.Run(fmt.Sprintf("Order %d", order), func(t *testing.T) {
			for litWidth := 6; litWidth <= 8; litWidth++ {
				t.Run(fmt.Sprintf("LitWidth %d", litWidth), func(t *testing.T) {
					var data []byte
					if litWidth == 6 {
						data = []byte{1, 2, 3}
					} else {
						data = []byte(`lorem ipsum dolor sit amet`)
					}
					var buf bytes.Buffer
					w := NewWriter(&buf, order, litWidth)
					if _, err := w.Write(data); err != nil {
						t.Errorf("write: %v: %v", string(data), err)
					}

					if err := w.Close(); err != nil {
						t.Errorf("close: %v", err)
					}

					b1 := buf.Bytes()
					buf.Reset()

					w.(*Writer).Reset(&buf, order, litWidth)

					if _, err := w.Write(data); err != nil {
						t.Errorf("write: %v: %v", string(data), err)
					}

					if err := w.Close(); err != nil {
						t.Errorf("close: %v", err)
					}
					b2 := buf.Bytes()

					if !bytes.Equal(b1, b2) {
						t.Errorf("bytes written were not same")
					}
				})
			}
		})
	}
}

func TestWriterReturnValues(t *testing.T) {
	w := NewWriter(io.Discard, LSB, 8)
	n, err := w.Write([]byte("asdf"))
	if n != 4 || err != nil {
		t.Errorf("got %d, %v, want 4, nil", n, err)
	}
}

func TestSmallLitWidth(t *testing.T) {
	w := NewWriter(io.Discard, LSB, 2)
	if _, err := w.Write([]byte{0x03}); err != nil {
		t.Fatalf("write a byte < 1<<2: %v", err)
	}
	if _, err := w.Write([]byte{0x04}); err == nil {
		t.Fatal("write a byte >= 1<<2: got nil error, want non-nil")
	}
}

func BenchmarkEncoder(b *testing.B) {
	buf, err := os.ReadFile("../testdata/e.txt")
	if err != nil {
		b.Fatal(err)
	}
	if len(buf) == 0 {
		b.Fatalf("test file has no data")
	}

	for e := 4; e <= 6; e++ {
		n := int(math.Pow10(e))
		buf0 := buf
		buf1 := make([]byte, n)
		for i := 0; i < n; i += len(buf0) {
			if len(buf0) > n-i {
				buf0 = buf0[:n-i]
			}
			copy(buf1[i:], buf0)
		}
		buf0 = nil
		runtime.GC()
		b.Run(fmt.Sprint("1e", e), func(b *testing.B) {
			b.SetBytes(int64(n))
			for i := 0; i < b.N; i++ {
				w := NewWriter(io.Discard, LSB, 8)
				w.Write(buf1)
				w.Close()
			}
		})
		b.Run(fmt.Sprint("1e-Reuse", e), func(b *testing.B) {
			b.SetBytes(int64(n))
			w := NewWriter(io.Discard, LSB, 8)
			for i := 0; i < b.N; i++ {
				w.Write(buf1)
				w.Close()
				w.(*Writer).Reset(io.Discard, LSB, 8)
			}
		})
	}
}
