// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build gofuzz

package png

import (
	"bytes"
	"fmt"
)

func Fuzz(data []byte) int {
	cfg, err := DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return 0
	}
	if cfg.Width*cfg.Height > 1e6 {
		return 0
	}
	img, err := Decode(bytes.NewReader(data))
	if err != nil {
		return 0
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
			panic(err)
		}
		img1, err := Decode(&w)
		if err != nil {
			panic(err)
		}
		got := img1.Bounds()
		want := img.Bounds()
		if !got.Eq(want) {
			fmt.Printf("bounds0: %#v\n", want)
			fmt.Printf("bounds1: %#v\n", got)
			panic("bounds have changed")
		}
	}
	return 1
}
