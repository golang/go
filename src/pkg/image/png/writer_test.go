// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package png

import (
	"fmt"
	"image"
	"io"
	"os"
	"testing"
)

func diff(m0, m1 image.Image) os.Error {
	if m0.Width() != m1.Width() || m0.Height() != m1.Height() {
		return os.NewError(fmt.Sprintf("dimensions differ: %dx%d vs %dx%d", m0.Width(), m0.Height(), m1.Width(), m1.Height()))
	}
	for y := 0; y < m0.Height(); y++ {
		for x := 0; x < m0.Width(); x++ {
			r0, g0, b0, a0 := m0.At(x, y).RGBA()
			r1, g1, b1, a1 := m1.At(x, y).RGBA()
			if r0 != r1 || g0 != g1 || b0 != b1 || a0 != a1 {
				return os.NewError(fmt.Sprintf("colors differ at (%d, %d): %v vs %v", x, y, m0.At(x, y), m1.At(x, y)))
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
