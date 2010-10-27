// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package png

import (
	"bytes"
	"fmt"
	"image"
	"io"
	"os"
	"testing"
)

func diff(m0, m1 image.Image) os.Error {
	b0, b1 := m0.Bounds(), m1.Bounds()
	if !b0.Eq(b1) {
		return fmt.Errorf("dimensions differ: %v vs %v", b0, b1)
	}
	for y := b0.Min.Y; y < b0.Max.Y; y++ {
		for x := b0.Min.X; x < b0.Max.X; x++ {
			r0, g0, b0, a0 := m0.At(x, y).RGBA()
			r1, g1, b1, a1 := m1.At(x, y).RGBA()
			if r0 != r1 || g0 != g1 || b0 != b1 || a0 != a1 {
				return fmt.Errorf("colors differ at (%d, %d): %v vs %v", x, y, m0.At(x, y), m1.At(x, y))
			}
		}
	}
	return nil
}

func TestWriter(t *testing.T) {
	// The filenames variable is declared in reader_test.go.
	for _, fn := range filenames {
		qfn := "testdata/pngsuite/" + fn + ".png"
		// Read the image.
		m0, err := readPng(qfn)
		if err != nil {
			t.Error(fn, err)
			continue
		}
		// Read the image again, and push it through a pipe that encodes at the write end, and decodes at the read end.
		pr, pw := io.Pipe()
		defer pr.Close()
		go func() {
			defer pw.Close()
			m1, err := readPng(qfn)
			if err != nil {
				t.Error(fn, err)
				return
			}
			err = Encode(pw, m1)
			if err != nil {
				t.Error(fn, err)
				return
			}
		}()
		m2, err := Decode(pr)
		if err != nil {
			t.Error(fn, err)
			continue
		}
		// Compare the two.
		err = diff(m0, m2)
		if err != nil {
			t.Error(fn, err)
			continue
		}
	}
}

func BenchmarkEncodePaletted(b *testing.B) {
	b.StopTimer()
	img := image.NewPaletted(640, 480,
		[]image.Color{
			image.RGBAColor{0, 0, 0, 255},
			image.RGBAColor{255, 255, 255, 255},
		})
	b.StartTimer()
	buffer := new(bytes.Buffer)
	for i := 0; i < b.N; i++ {
		buffer.Reset()
		Encode(buffer, img)
	}
}
