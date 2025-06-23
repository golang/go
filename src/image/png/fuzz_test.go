// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package png

import (
	"bytes"
	"image"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func FuzzDecode(f *testing.F) {
	if testing.Short() {
		f.Skip("Skipping in short mode")
	}

	testdata, err := os.ReadDir("../testdata")
	if err != nil {
		f.Fatalf("failed to read testdata directory: %s", err)
	}
	for _, de := range testdata {
		if de.IsDir() || !strings.HasSuffix(de.Name(), ".png") {
			continue
		}
		b, err := os.ReadFile(filepath.Join("../testdata", de.Name()))
		if err != nil {
			f.Fatalf("failed to read testdata: %s", err)
		}
		f.Add(b)
	}

	f.Fuzz(func(t *testing.T, b []byte) {
		cfg, _, err := image.DecodeConfig(bytes.NewReader(b))
		if err != nil {
			return
		}
		if cfg.Width*cfg.Height > 1e6 {
			return
		}
		img, typ, err := image.Decode(bytes.NewReader(b))
		if err != nil || typ != "png" {
			return
		}
		levels := []CompressionLevel{
			DefaultCompression,
			NoCompression,
			BestSpeed,
			BestCompression,
		}
		for _, l := range levels {
			var w bytes.Buffer
			e := &Encoder{CompressionLevel: l}
			err = e.Encode(&w, img)
			if err != nil {
				t.Errorf("failed to encode valid image: %s", err)
				continue
			}
			img1, err := Decode(&w)
			if err != nil {
				t.Errorf("failed to decode roundtripped image: %s", err)
				continue
			}
			got := img1.Bounds()
			want := img.Bounds()
			if !got.Eq(want) {
				t.Errorf("roundtripped image bounds have changed, got: %s, want: %s", got, want)
			}
		}
	})
}
