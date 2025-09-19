// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package saferio

import (
	"bytes"
	"io"
	"math"
	"testing"
)

func TestReadData(t *testing.T) {
	const count = 100
	input := bytes.Repeat([]byte{'a'}, count)

	t.Run("small", func(t *testing.T) {
		got, err := ReadData(bytes.NewReader(input), count)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, input) {
			t.Errorf("got %v, want %v", got, input)
		}
	})

	t.Run("large", func(t *testing.T) {
		_, err := ReadData(bytes.NewReader(input), 10<<30)
		if err == nil {
			t.Error("large read succeeded unexpectedly")
		}
	})

	t.Run("maxint", func(t *testing.T) {
		_, err := ReadData(bytes.NewReader(input), 1<<62)
		if err == nil {
			t.Error("large read succeeded unexpectedly")
		}
	})

	t.Run("small-EOF", func(t *testing.T) {
		_, err := ReadData(bytes.NewReader(nil), chunk-1)
		if err != io.EOF {
			t.Errorf("ReadData = %v, want io.EOF", err)
		}
	})

	t.Run("large-EOF", func(t *testing.T) {
		_, err := ReadData(bytes.NewReader(nil), chunk+1)
		if err != io.EOF {
			t.Errorf("ReadData = %v, want io.EOF", err)
		}
	})

	t.Run("large-UnexpectedEOF", func(t *testing.T) {
		_, err := ReadData(bytes.NewReader(make([]byte, chunk)), chunk+1)
		if err != io.ErrUnexpectedEOF {
			t.Errorf("ReadData = %v, want io.ErrUnexpectedEOF", err)
		}
	})
}

func TestReadDataAt(t *testing.T) {
	const count = 100
	input := bytes.Repeat([]byte{'a'}, count)

	t.Run("small", func(t *testing.T) {
		got, err := ReadDataAt(bytes.NewReader(input), count, 0)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, input) {
			t.Errorf("got %v, want %v", got, input)
		}
	})

	t.Run("large", func(t *testing.T) {
		_, err := ReadDataAt(bytes.NewReader(input), 10<<30, 0)
		if err == nil {
			t.Error("large read succeeded unexpectedly")
		}
	})

	t.Run("maxint", func(t *testing.T) {
		_, err := ReadDataAt(bytes.NewReader(input), 1<<62, 0)
		if err == nil {
			t.Error("large read succeeded unexpectedly")
		}
	})

	t.Run("SectionReader", func(t *testing.T) {
		// Reading 0 bytes from an io.SectionReader at the end
		// of the section will return EOF, but ReadDataAt
		// should succeed and return 0 bytes.
		sr := io.NewSectionReader(bytes.NewReader(input), 0, 0)
		got, err := ReadDataAt(sr, 0, 0)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) > 0 {
			t.Errorf("got %d bytes, expected 0", len(got))
		}
	})
}

func TestSliceCap(t *testing.T) {
	t.Run("small", func(t *testing.T) {
		c := SliceCap[int](10)
		if c != 10 {
			t.Errorf("got capacity %d, want %d", c, 10)
		}
	})

	t.Run("large", func(t *testing.T) {
		c := SliceCap[byte](1 << 30)
		if c < 0 {
			t.Error("SliceCap failed unexpectedly")
		} else if c == 1<<30 {
			t.Errorf("got capacity %d which is too high", c)
		}
	})

	t.Run("maxint", func(t *testing.T) {
		c := SliceCap[byte](1 << 63)
		if c >= 0 {
			t.Errorf("SliceCap returned %d, expected failure", c)
		}
	})

	t.Run("overflow", func(t *testing.T) {
		c := SliceCap[int64](1 << 62)
		if c >= 0 {
			t.Errorf("SliceCap returned %d, expected failure", c)
		}
	})
}

func TestInBounds32(t *testing.T) {
	tests := []struct {
		name   string
		slice  []struct{}
		start  uint32
		length uint32
		want   bool
	}{
		{"valid range", make([]struct{}, 5), 1, 3, true},
		{"start+length equals len", make([]struct{}, 3), 0, 3, false},
		{"start+length exceeds len", make([]struct{}, 3), 2, 2, false},
		{"start at end", make([]struct{}, 3), 3, 0, false},
		{"zero length", make([]struct{}, 3), 1, 0, true},
		{"empty slice", make([]struct{}, 0), 0, 0, false},
		{"maxuint32 overflow", make([]struct{}, 3), math.MaxUint32, 1, false},
		{"maxuint32 no overflow", make([]struct{}, 3), 0, math.MaxUint32, false},
		{"maxuint32 edge", make([]struct{}, 3), math.MaxUint32 - 1, 1, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := InBounds32(tt.slice, tt.start, tt.length)
			if got != tt.want {
				t.Errorf("InBounds32(%v, %d, %d) = %v, want %v", len(tt.slice), tt.start, tt.length, got, tt.want)
			}
		})
	}
}

func TestInBounds64(t *testing.T) {
	tests := []struct {
		name   string
		slice  []struct{}
		start  uint64
		length uint64
		want   bool
	}{
		{"valid range", make([]struct{}, 5), 1, 3, true},
		{"start+length equals len", make([]struct{}, 3), 0, 3, false},
		{"start+length exceeds len", make([]struct{}, 3), 2, 2, false},
		{"start at end", make([]struct{}, 3), 3, 0, false},
		{"zero length", make([]struct{}, 3), 1, 0, true},
		{"empty slice", make([]struct{}, 0), 0, 0, false},
		{"maxuint64 overflow", make([]struct{}, 3), math.MaxUint64, 1, false},
		{"maxuint64 no overflow", make([]struct{}, 3), 0, math.MaxUint64, false},
		{"maxuint64 edge", make([]struct{}, 3), math.MaxUint64 - 1, 1, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := InBounds64(tt.slice, tt.start, tt.length)
			if got != tt.want {
				t.Errorf("InBounds64(%v, %d, %d) = %v, want %v", tt.slice, tt.start, tt.length, got, tt.want)
			}
		})
	}
}
