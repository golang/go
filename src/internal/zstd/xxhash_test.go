// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package zstd

import (
	"bytes"
	"os"
	"os/exec"
	"strconv"
	"testing"
)

var xxHashTests = []struct {
	data string
	hash uint64
}{
	{
		"hello, world",
		0xb33a384e6d1b1242,
	},
	{
		"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789$",
		0x1032d841e824f998,
	},
}

func TestXXHash(t *testing.T) {
	var xh xxhash64
	for i, test := range xxHashTests {
		xh.reset()
		xh.update([]byte(test.data))
		if got := xh.digest(); got != test.hash {
			t.Errorf("#%d: got %#x want %#x", i, got, test.hash)
		}
	}
}

func TestLargeXXHash(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping expensive test in short mode")
	}

	data, err := os.ReadFile("../../testdata/Isaac.Newton-Opticks.txt")
	if err != nil {
		t.Fatal(err)
	}

	var xh xxhash64
	xh.reset()
	i := 0
	for i < len(data) {
		// Write varying amounts to test buffering.
		c := i%4094 + 1
		if i+c > len(data) {
			c = len(data) - i
		}
		xh.update(data[i : i+c])
		i += c
	}

	got := xh.digest()
	want := uint64(0xf0dd39fd7e063f82)
	if got != want {
		t.Errorf("got %#x want %#x", got, want)
	}
}

func findXxhsum(t testing.TB) string {
	xxhsum, err := exec.LookPath("xxhsum")
	if err != nil {
		t.Skip("skipping because xxhsum not found")
	}
	return xxhsum
}

func FuzzXXHash(f *testing.F) {
	xxhsum := findXxhsum(f)

	for _, test := range xxHashTests {
		f.Add([]byte(test.data))
	}
	f.Add(bytes.Repeat([]byte("abcdefghijklmnop"), 256))
	var buf bytes.Buffer
	for i := 0; i < 256; i++ {
		buf.WriteByte(byte(i))
	}
	f.Add(bytes.Repeat(buf.Bytes(), 64))
	f.Add(bigData(f))

	f.Fuzz(func { t, b ->
		cmd := exec.Command(xxhsum, "-H64")
		cmd.Stdin = bytes.NewReader(b)
		var hhsumHash bytes.Buffer
		cmd.Stdout = &hhsumHash
		if err := cmd.Run(); err != nil {
			t.Fatalf("running hhsum failed: %v", err)
		}
		hhHashBytes := bytes.Fields(bytes.TrimSpace(hhsumHash.Bytes()))[0]
		hhHash, err := strconv.ParseUint(string(hhHashBytes), 16, 64)
		if err != nil {
			t.Fatalf("could not parse hash %q: %v", hhHashBytes, err)
		}

		var xh xxhash64
		xh.reset()
		xh.update(b)
		goHash := xh.digest()

		if goHash != hhHash {
			t.Errorf("Go hash %#x != xxhsum hash %#x", goHash, hhHash)
		}
	})
}
