// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package zstd

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"internal/race"
	"internal/testenv"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// tests holds some simple test cases, including some found by fuzzing.
var tests = []struct {
	name, uncompressed, compressed string
}{
	{
		"hello",
		"hello, world\n",
		"\x28\xb5\x2f\xfd\x24\x0d\x69\x00\x00\x68\x65\x6c\x6c\x6f\x2c\x20\x77\x6f\x72\x6c\x64\x0a\x4c\x1f\xf9\xf1",
	},
	{
		// a small compressed .debug_ranges section.
		"ranges",
		"\xcc\x11\x00\x00\x00\x00\x00\x00\xd5\x13\x00\x00\x00\x00\x00\x00" +
			"\x1c\x14\x00\x00\x00\x00\x00\x00\x72\x14\x00\x00\x00\x00\x00\x00" +
			"\x9d\x14\x00\x00\x00\x00\x00\x00\xd5\x14\x00\x00\x00\x00\x00\x00" +
			"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00" +
			"\xfb\x12\x00\x00\x00\x00\x00\x00\x09\x13\x00\x00\x00\x00\x00\x00" +
			"\x0c\x13\x00\x00\x00\x00\x00\x00\xcb\x13\x00\x00\x00\x00\x00\x00" +
			"\x29\x14\x00\x00\x00\x00\x00\x00\x4e\x14\x00\x00\x00\x00\x00\x00" +
			"\x9d\x14\x00\x00\x00\x00\x00\x00\xd5\x14\x00\x00\x00\x00\x00\x00" +
			"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00" +
			"\xfb\x12\x00\x00\x00\x00\x00\x00\x09\x13\x00\x00\x00\x00\x00\x00" +
			"\x67\x13\x00\x00\x00\x00\x00\x00\xcb\x13\x00\x00\x00\x00\x00\x00" +
			"\x9d\x14\x00\x00\x00\x00\x00\x00\xd5\x14\x00\x00\x00\x00\x00\x00" +
			"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00" +
			"\x5f\x0b\x00\x00\x00\x00\x00\x00\x6c\x0b\x00\x00\x00\x00\x00\x00" +
			"\x7d\x0b\x00\x00\x00\x00\x00\x00\x7e\x0c\x00\x00\x00\x00\x00\x00" +
			"\x38\x0f\x00\x00\x00\x00\x00\x00\x5c\x0f\x00\x00\x00\x00\x00\x00" +
			"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00" +
			"\x83\x0c\x00\x00\x00\x00\x00\x00\xfa\x0c\x00\x00\x00\x00\x00\x00" +
			"\xfd\x0d\x00\x00\x00\x00\x00\x00\xef\x0e\x00\x00\x00\x00\x00\x00" +
			"\x14\x0f\x00\x00\x00\x00\x00\x00\x38\x0f\x00\x00\x00\x00\x00\x00" +
			"\x9f\x0f\x00\x00\x00\x00\x00\x00\xac\x0f\x00\x00\x00\x00\x00\x00" +
			"\xdb\x0f\x00\x00\x00\x00\x00\x00\xff\x0f\x00\x00\x00\x00\x00\x00" +
			"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00" +
			"\xfd\x0d\x00\x00\x00\x00\x00\x00\xd8\x0e\x00\x00\x00\x00\x00\x00" +
			"\x9f\x0f\x00\x00\x00\x00\x00\x00\xac\x0f\x00\x00\x00\x00\x00\x00" +
			"\xdb\x0f\x00\x00\x00\x00\x00\x00\xff\x0f\x00\x00\x00\x00\x00\x00" +
			"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00" +
			"\xfa\x0c\x00\x00\x00\x00\x00\x00\xea\x0d\x00\x00\x00\x00\x00\x00" +
			"\xef\x0e\x00\x00\x00\x00\x00\x00\x14\x0f\x00\x00\x00\x00\x00\x00" +
			"\x5c\x0f\x00\x00\x00\x00\x00\x00\x9f\x0f\x00\x00\x00\x00\x00\x00" +
			"\xac\x0f\x00\x00\x00\x00\x00\x00\xdb\x0f\x00\x00\x00\x00\x00\x00" +
			"\xff\x0f\x00\x00\x00\x00\x00\x00\x2c\x10\x00\x00\x00\x00\x00\x00" +
			"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00" +
			"\x60\x11\x00\x00\x00\x00\x00\x00\xd1\x16\x00\x00\x00\x00\x00\x00" +
			"\x40\x0b\x00\x00\x00\x00\x00\x00\x2c\x10\x00\x00\x00\x00\x00\x00" +
			"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00" +
			"\x7a\x00\x00\x00\x00\x00\x00\x00\xb6\x00\x00\x00\x00\x00\x00\x00" +
			"\x9f\x01\x00\x00\x00\x00\x00\x00\xa7\x01\x00\x00\x00\x00\x00\x00" +
			"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00" +
			"\x7a\x00\x00\x00\x00\x00\x00\x00\xa9\x00\x00\x00\x00\x00\x00\x00" +
			"\x9f\x01\x00\x00\x00\x00\x00\x00\xa7\x01\x00\x00\x00\x00\x00\x00" +
			"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",

		"\x28\xb5\x2f\xfd\x64\xa0\x01\x2d\x05\x00\xc4\x04\xcc\x11\x00\xd5" +
			"\x13\x00\x1c\x14\x00\x72\x9d\xd5\xfb\x12\x00\x09\x0c\x13\xcb\x13" +
			"\x29\x4e\x67\x5f\x0b\x6c\x0b\x7d\x0b\x7e\x0c\x38\x0f\x5c\x0f\x83" +
			"\x0c\xfa\x0c\xfd\x0d\xef\x0e\x14\x38\x9f\x0f\xac\x0f\xdb\x0f\xff" +
			"\x0f\xd8\x9f\xac\xdb\xff\xea\x5c\x2c\x10\x60\xd1\x16\x40\x0b\x7a" +
			"\x00\xb6\x00\x9f\x01\xa7\x01\xa9\x36\x20\xa0\x83\x14\x34\x63\x4a" +
			"\x21\x70\x8c\x07\x46\x03\x4e\x10\x62\x3c\x06\x4e\xc8\x8c\xb0\x32" +
			"\x2a\x59\xad\xb2\xf1\x02\x82\x7c\x33\xcb\x92\x6f\x32\x4f\x9b\xb0" +
			"\xa2\x30\xf0\xc0\x06\x1e\x98\x99\x2c\x06\x1e\xd8\xc0\x03\x56\xd8" +
			"\xc0\x03\x0f\x6c\xe0\x01\xf1\xf0\xee\x9a\xc6\xc8\x97\x99\xd1\x6c" +
			"\xb4\x21\x45\x3b\x10\xe4\x7b\x99\x4d\x8a\x36\x64\x5c\x77\x08\x02" +
			"\xcb\xe0\xce",
	},
	{
		"fuzz1",
		"0\x00\x00\x00\x00\x000\x00\x00\x00\x00\x001\x00\x00\x00\x00\x000000",
		"(\xb5/\xfd\x04X\x8d\x00\x00P0\x000\x001\x000000\x03T\x02\x00\x01\x01m\xf9\xb7G",
	},
	{
		"empty block",
		"",
		"\x28\xb5\x2f\xfd\x00\x00\x15\x00\x00\x00\x00",
	},
	{
		"single skippable frame",
		"",
		"\x50\x2a\x4d\x18\x00\x00\x00\x00",
	},
	{
		"two skippable frames",
		"",
		"\x50\x2a\x4d\x18\x00\x00\x00\x00" +
			"\x50\x2a\x4d\x18\x00\x00\x00\x00",
	},
}

func TestSamples(t *testing.T) {
	for _, test := range tests {
		test := test
		t.Run(test.name, func { t ->
			r := NewReader(strings.NewReader(test.compressed))
			got, err := io.ReadAll(r)
			if err != nil {
				t.Fatal(err)
			}
			gotstr := string(got)
			if gotstr != test.uncompressed {
				t.Errorf("got %q want %q", gotstr, test.uncompressed)
			}
		})
	}
}

func TestReset(t *testing.T) {
	input := strings.NewReader("")
	r := NewReader(input)
	for _, test := range tests {
		test := test
		t.Run(test.name, func { t ->
			input.Reset(test.compressed)
			r.Reset(input)
			got, err := io.ReadAll(r)
			if err != nil {
				t.Fatal(err)
			}
			gotstr := string(got)
			if gotstr != test.uncompressed {
				t.Errorf("got %q want %q", gotstr, test.uncompressed)
			}
		})
	}
}

var (
	bigDataOnce  sync.Once
	bigDataBytes []byte
	bigDataErr   error
)

// bigData returns the contents of our large test file repeated multiple times.
func bigData(t testing.TB) []byte {
	bigDataOnce.Do(func() {
		bigDataBytes, bigDataErr = os.ReadFile("../../testdata/Isaac.Newton-Opticks.txt")
		if bigDataErr == nil {
			bigDataBytes = bytes.Repeat(bigDataBytes, 20)
		}
	})
	if bigDataErr != nil {
		t.Fatal(bigDataErr)
	}
	return bigDataBytes
}

func findZstd(t testing.TB) string {
	zstd, err := exec.LookPath("zstd")
	if err != nil {
		t.Skip("skipping because zstd not found")
	}
	return zstd
}

var (
	zstdBigOnce  sync.Once
	zstdBigBytes []byte
	zstdBigErr   error
)

// zstdBigData returns the compressed contents of our large test file.
// This will only run on Unix systems with zstd installed.
// That's OK as the package is GOOS-independent.
func zstdBigData(t testing.TB) []byte {
	input := bigData(t)

	zstd := findZstd(t)

	zstdBigOnce.Do(func() {
		cmd := exec.Command(zstd, "-z")
		cmd.Stdin = bytes.NewReader(input)
		var compressed bytes.Buffer
		cmd.Stdout = &compressed
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			zstdBigErr = fmt.Errorf("running zstd failed: %v", err)
			return
		}

		zstdBigBytes = compressed.Bytes()
	})
	if zstdBigErr != nil {
		t.Fatal(zstdBigErr)
	}
	return zstdBigBytes
}

// Test decompressing a large file. We don't have a compressor,
// so this test only runs on systems with zstd installed.
func TestLarge(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping expensive test in short mode")
	}

	data := bigData(t)
	compressed := zstdBigData(t)

	t.Logf("zstd compressed %d bytes to %d", len(data), len(compressed))

	r := NewReader(bytes.NewReader(compressed))
	got, err := io.ReadAll(r)
	if err != nil {
		t.Fatal(err)
	}

	if !bytes.Equal(got, data) {
		showDiffs(t, got, data)
	}
}

// showDiffs reports the first few differences in two []byte.
func showDiffs(t *testing.T, got, want []byte) {
	t.Error("data mismatch")
	if len(got) != len(want) {
		t.Errorf("got data length %d, want %d", len(got), len(want))
	}
	diffs := 0
	for i, b := range got {
		if i >= len(want) {
			break
		}
		if b != want[i] {
			diffs++
			if diffs > 20 {
				break
			}
			t.Logf("%d: %#x != %#x", i, b, want[i])
		}
	}
}

func TestAlloc(t *testing.T) {
	testenv.SkipIfOptimizationOff(t)
	if race.Enabled {
		t.Skip("skipping allocation test under race detector")
	}

	compressed := zstdBigData(t)
	input := bytes.NewReader(compressed)
	r := NewReader(input)
	c := testing.AllocsPerRun(10, func() {
		input.Reset(compressed)
		r.Reset(input)
		io.Copy(io.Discard, r)
	})
	if c != 0 {
		t.Errorf("got %v allocs, want 0", c)
	}
}

func TestFileSamples(t *testing.T) {
	samples, err := os.ReadDir("testdata")
	if err != nil {
		t.Fatal(err)
	}

	for _, sample := range samples {
		name := sample.Name()
		if !strings.HasSuffix(name, ".zst") {
			continue
		}

		t.Run(name, func { t ->
			f, err := os.Open(filepath.Join("testdata", name))
			if err != nil {
				t.Fatal(err)
			}

			r := NewReader(f)
			h := sha256.New()
			if _, err := io.Copy(h, r); err != nil {
				t.Fatal(err)
			}
			got := fmt.Sprintf("%x", h.Sum(nil))[:8]

			want, _, _ := strings.Cut(name, ".")
			if got != want {
				t.Errorf("Wrong uncompressed content hash: got %s, want %s", got, want)
			}
		})
	}
}

func TestReaderBad(t *testing.T) {
	for i, s := range badStrings {
		t.Run(fmt.Sprintf("badStrings#%d", i), func { t ->
			_, err := io.Copy(io.Discard, NewReader(strings.NewReader(s)))
			if err == nil {
				t.Error("expected error")
			}
		})
	}
}

func BenchmarkLarge(b *testing.B) {
	b.StopTimer()
	b.ReportAllocs()

	compressed := zstdBigData(b)

	b.SetBytes(int64(len(compressed)))

	input := bytes.NewReader(compressed)
	r := NewReader(input)

	b.StartTimer()
	for i := 0; i < b.N; i++ {
		input.Reset(compressed)
		r.Reset(input)
		io.Copy(io.Discard, r)
	}
}
