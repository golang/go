// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package zstd

import (
	"bytes"
	"io"
	"os"
	"os/exec"
	"testing"
)

// badStrings is some inputs that FuzzReader failed on earlier.
var badStrings = []string{
	"(\xb5/\xfdd00,\x05\x00\xc4\x0400000000000000000000000000000000000000000000000000000000000000000000000000000 \xa07100000000000000000000000000000000000000000000000000000000000000000000000000aM\x8a2y0B\b",
	"(\xb5/\xfd00$\x05\x0020 00X70000a70000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
	"(\xb5/\xfd00$\x05\x0020 00B00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
	"(\xb5/\xfd00}\x00\x0020\x00\x9000000000000",
	"(\xb5/\xfd00}\x00\x00&0\x02\x830!000000000",
	"(\xb5/\xfd\x1002000$\x05\x0010\xcc0\xa8100000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
	"(\xb5/\xfd\x1002000$\x05\x0000\xcc0\xa8100d\x0000001000000000000000000000000000000000000000000000000000000000000000000000000\x000000000000000000000000000000000000000000000000000000000000000000000000000000",
	"(\xb5/\xfd001\x00\x0000000000000000000",
	"(\xb5/\xfd00\xec\x00\x00&@\x05\x05A7002\x02\x00\x02\x00\x02\x0000000000000000",
	"(\xb5/\xfd00\xec\x00\x00V@\x05\x0517002\x02\x00\x02\x00\x02\x0000000000000000",
	"\x50\x2a\x4d\x18\x02\x00\x00\x00",
	"(\xb5/\xfd\xe40000000\xfa20\x000",
}

// This is a simple fuzzer to see if the decompressor panics.
func FuzzReader(f *testing.F) {
	for _, test := range tests {
		f.Add([]byte(test.compressed))
	}
	for _, s := range badStrings {
		f.Add([]byte(s))
	}
	f.Fuzz(func(t *testing.T, b []byte) {
		r := NewReader(bytes.NewReader(b))
		io.Copy(io.Discard, r)
	})
}

// Fuzz test to verify that what we decompress is what we compress.
// This isn't a great fuzz test because the fuzzer can't efficiently
// explore the space of decompressor behavior, since it can't see
// what the compressor is doing. But it's better than nothing.
func FuzzDecompressor(f *testing.F) {
	zstd := findZstd(f)

	for _, test := range tests {
		f.Add([]byte(test.uncompressed))
	}

	// Add some larger data, as that has more interesting compression.
	f.Add(bytes.Repeat([]byte("abcdefghijklmnop"), 256))
	var buf bytes.Buffer
	for i := 0; i < 256; i++ {
		buf.WriteByte(byte(i))
	}
	f.Add(bytes.Repeat(buf.Bytes(), 64))
	f.Add(bigData(f))

	f.Fuzz(func(t *testing.T, b []byte) {
		cmd := exec.Command(zstd, "-z")
		cmd.Stdin = bytes.NewReader(b)
		var compressed bytes.Buffer
		cmd.Stdout = &compressed
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			t.Errorf("running zstd failed: %v", err)
		}

		r := NewReader(bytes.NewReader(compressed.Bytes()))
		got, err := io.ReadAll(r)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, b) {
			showDiffs(t, got, b)
		}
	})
}

// Fuzz test to check that if we can decompress some data,
// so can zstd, and that we get the same result.
func FuzzReverse(f *testing.F) {
	zstd := findZstd(f)

	for _, test := range tests {
		f.Add([]byte(test.compressed))
	}

	// Set a hook to reject some cases where we don't match zstd.
	fuzzing = true
	defer func() { fuzzing = false }()

	f.Fuzz(func(t *testing.T, b []byte) {
		r := NewReader(bytes.NewReader(b))
		goExp, goErr := io.ReadAll(r)

		cmd := exec.Command(zstd, "-d")
		cmd.Stdin = bytes.NewReader(b)
		var uncompressed bytes.Buffer
		cmd.Stdout = &uncompressed
		cmd.Stderr = os.Stderr
		zstdErr := cmd.Run()
		zstdExp := uncompressed.Bytes()

		if goErr == nil && zstdErr == nil {
			if !bytes.Equal(zstdExp, goExp) {
				showDiffs(t, zstdExp, goExp)
			}
		} else {
			// Ideally we should check that this package and
			// the zstd program both fail or both succeed,
			// and that if they both fail one byte sequence
			// is an exact prefix of the other.
			// Actually trying this proved to be frustrating,
			// as the zstd program appears to accept invalid
			// byte sequences using rules that are difficult
			// to determine.
			// So we just check the prefix.

			c := len(goExp)
			if c > len(zstdExp) {
				c = len(zstdExp)
			}
			goExp = goExp[:c]
			zstdExp = zstdExp[:c]
			if !bytes.Equal(goExp, zstdExp) {
				t.Error("byte mismatch after error")
				t.Logf("Go error: %v\n", goErr)
				t.Logf("zstd error: %v\n", zstdErr)
				showDiffs(t, zstdExp, goExp)
			}
		}
	})
}
