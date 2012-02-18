// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This example demonstrates decoding a JPEG image and examining its pixels.
package image_test

import (
	"fmt"
	"image"
	"log"
	"os"

	// Package image/jpeg is not used explicitly in the code below,
	// but is imported for its initialization side-effect, which allows
	// image.Decode to understand JPEG formatted images. Uncomment these
	// two lines to also understand GIF and PNG images:
	// _ "image/gif"
	// _ "image/png"
	_ "image/jpeg"
)

func Example() {
	// Open the file.
	file, err := os.Open("testdata/video-001.jpeg")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	// Decode the image.
	m, _, err := image.Decode(file)
	if err != nil {
		log.Fatal(err)
	}
	bounds := m.Bounds()

	// Calculate a 16-bin histogram for m's red, green, blue and alpha components.
	//
	// An image's bounds do not necessarily start at (0, 0), so the two loops start
	// at bounds.Min.Y and bounds.Min.X. Looping over Y first and X second is more
	// likely to result in better memory access patterns than X first and Y second.
	var histogram [16][4]int
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, a := m.At(x, y).RGBA()
			// A color's RGBA method returns values in the range [0, 65535].
			// Shifting by 12 reduces this to the range [0, 15].
			histogram[r>>12][0]++
			histogram[g>>12][1]++
			histogram[b>>12][2]++
			histogram[a>>12][3]++
		}
	}

	// Print the results.
	fmt.Printf("%-14s %6s %6s %6s %6s\n", "bin", "red", "green", "blue", "alpha")
	for i, x := range histogram {
		fmt.Printf("0x%04x-0x%04x: %6d %6d %6d %6d\n", i<<12, (i+1)<<12-1, x[0], x[1], x[2], x[3])
	}
	// Output:
	// bin               red  green   blue  alpha
	// 0x0000-0x0fff:    471    819   7596      0
	// 0x1000-0x1fff:    576   2892    726      0
	// 0x2000-0x2fff:   1038   2330    943      0
	// 0x3000-0x3fff:    883   2321   1014      0
	// 0x4000-0x4fff:    501   1295    525      0
	// 0x5000-0x5fff:    302    962    242      0
	// 0x6000-0x6fff:    219    358    150      0
	// 0x7000-0x7fff:    352    281    192      0
	// 0x8000-0x8fff:   3688    216    246      0
	// 0x9000-0x9fff:   2277    237    283      0
	// 0xa000-0xafff:    971    254    357      0
	// 0xb000-0xbfff:    317    306    429      0
	// 0xc000-0xcfff:    203    402    401      0
	// 0xd000-0xdfff:    256    394    241      0
	// 0xe000-0xefff:    378    343    173      0
	// 0xf000-0xffff:   3018   2040   1932  15450
}
