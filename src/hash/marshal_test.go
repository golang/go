// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that the hashes in the standard library implement
// BinaryMarshaler, BinaryUnmarshaler,
// and lock in the current representations.

package hash_test

import (
	"bytes"
	"crypto/md5"
	"crypto/sha1"
	"crypto/sha256"
	"crypto/sha512"
	"encoding"
	"encoding/hex"
	"hash"
	"hash/adler32"
	"hash/crc32"
	"hash/crc64"
	"hash/fnv"
	"testing"
)

func fromHex(s string) []byte {
	b, err := hex.DecodeString(s)
	if err != nil {
		panic(err)
	}
	return b
}

var marshalTests = []struct {
	name   string
	new    func() hash.Hash
	golden []byte
}{
	{"adler32", func() hash.Hash { return adler32.New() }, fromHex("61646c01460a789d")},
	{"crc32", func() hash.Hash { return crc32.NewIEEE() }, fromHex("63726301ca87914dc956d3e8")},
	{"crc64", func() hash.Hash { return crc64.New(crc64.MakeTable(crc64.ISO)) }, fromHex("6372630273ba8484bbcd5def5d51c83c581695be")},
	{"fnv32", func() hash.Hash { return fnv.New32() }, fromHex("666e760171ba3d77")},
	{"fnv32a", func() hash.Hash { return fnv.New32a() }, fromHex("666e76027439f86f")},
	{"fnv64", func() hash.Hash { return fnv.New64() }, fromHex("666e7603cc64e0e97692c637")},
	{"fnv64a", func() hash.Hash { return fnv.New64a() }, fromHex("666e7604c522af9b0dede66f")},
	{"fnv128", func() hash.Hash { return fnv.New128() }, fromHex("666e760561587a70a0f66d7981dc980e2cabbaf7")},
	{"fnv128a", func() hash.Hash { return fnv.New128a() }, fromHex("666e7606a955802b0136cb67622b461d9f91e6ff")},
	{"md5", md5.New, fromHex("6d643501a91b0023007aa14740a3979210b5f024c0c1c2c3c4c5c6c7c8c9cacbcccdcecfd0d1d2d3d4d5d6d7d8d9dadbdcdddedfe0e1e2e3e4e5e6e7e8e9eaebecedeeeff0f1f2f3f4f5f6f7f80000000000000000000000000000f9")},
	{"sha1", sha1.New, fromHex("736861016dad5acb4dc003952f7a0b352ee5537ec381a228c0c1c2c3c4c5c6c7c8c9cacbcccdcecfd0d1d2d3d4d5d6d7d8d9dadbdcdddedfe0e1e2e3e4e5e6e7e8e9eaebecedeeeff0f1f2f3f4f5f6f7f80000000000000000000000000000f9")},
	{"sha224", sha256.New224, fromHex("73686102f8b92fc047c9b4d82f01a6370841277b7a0d92108440178c83db855a8e66c2d9c0c1c2c3c4c5c6c7c8c9cacbcccdcecfd0d1d2d3d4d5d6d7d8d9dadbdcdddedfe0e1e2e3e4e5e6e7e8e9eaebecedeeeff0f1f2f3f4f5f6f7f80000000000000000000000000000f9")},
	{"sha256", sha256.New, fromHex("736861032bed68b99987cae48183b2b049d393d0050868e4e8ba3730e9112b08765929b7c0c1c2c3c4c5c6c7c8c9cacbcccdcecfd0d1d2d3d4d5d6d7d8d9dadbdcdddedfe0e1e2e3e4e5e6e7e8e9eaebecedeeeff0f1f2f3f4f5f6f7f80000000000000000000000000000f9")},
	{"sha384", sha512.New384, fromHex("736861046f1664d213dd802f7c47bc50637cf93592570a2b8695839148bf38341c6eacd05326452ef1cbe64d90f1ef73bb5ac7d2803565467d0ddb10c5ee3fc050f9f0c1808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9fa0a1a2a3a4a5a6a7a8a9aaabacadaeafb0b1b2b3b4b5b6b7b8b9babbbcbdbebfc0c1c2c3c4c5c6c7c8c9cacbcccdcecfd0d1d2d3d4d5d6d7d8d9dadbdcdddedfe0e1e2e3e4e5e6e7e8e9eaebecedeeeff0f1f2f3f4f5f6f7f80000000000000000000000000000f9")},
	{"sha512_224", sha512.New512_224, fromHex("736861056f1a450ec15af20572d0d1ee6518104d7cbbbe79a038557af5450ed7dbd420b53b7335209e951b4d9aff401f90549b9604fa3d823fbb8581c73582a88aa84022808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9fa0a1a2a3a4a5a6a7a8a9aaabacadaeafb0b1b2b3b4b5b6b7b8b9babbbcbdbebfc0c1c2c3c4c5c6c7c8c9cacbcccdcecfd0d1d2d3d4d5d6d7d8d9dadbdcdddedfe0e1e2e3e4e5e6e7e8e9eaebecedeeeff0f1f2f3f4f5f6f7f80000000000000000000000000000f9")},
	{"sha512_256", sha512.New512_256, fromHex("736861067c541f1d1a72536b1f5dad64026bcc7c508f8a2126b51f46f8b9bff63a26fee70980718031e96832e95547f4fe76160ff84076db53b4549b86354af8e17b5116808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9fa0a1a2a3a4a5a6a7a8a9aaabacadaeafb0b1b2b3b4b5b6b7b8b9babbbcbdbebfc0c1c2c3c4c5c6c7c8c9cacbcccdcecfd0d1d2d3d4d5d6d7d8d9dadbdcdddedfe0e1e2e3e4e5e6e7e8e9eaebecedeeeff0f1f2f3f4f5f6f7f80000000000000000000000000000f9")},
	{"sha512", sha512.New, fromHex("736861078e03953cd57cd6879321270afa70c5827bb5b69be59a8f0130147e94f2aedf7bdc01c56c92343ca8bd837bb7f0208f5a23e155694516b6f147099d491a30b151808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9fa0a1a2a3a4a5a6a7a8a9aaabacadaeafb0b1b2b3b4b5b6b7b8b9babbbcbdbebfc0c1c2c3c4c5c6c7c8c9cacbcccdcecfd0d1d2d3d4d5d6d7d8d9dadbdcdddedfe0e1e2e3e4e5e6e7e8e9eaebecedeeeff0f1f2f3f4f5f6f7f80000000000000000000000000000f9")},
}

func TestMarshalHash(t *testing.T) {
	for _, tt := range marshalTests {
		t.Run(tt.name, func(t *testing.T) {
			buf := make([]byte, 256)
			for i := range buf {
				buf[i] = byte(i)
			}

			h := tt.new()
			h.Write(buf[:256])
			sum := h.Sum(nil)

			h2 := tt.new()
			h3 := tt.new()
			const split = 249
			for i := 0; i < split; i++ {
				h2.Write(buf[i : i+1])
			}
			h2m, ok := h2.(encoding.BinaryMarshaler)
			if !ok {
				t.Fatalf("Hash does not implement MarshalBinary")
			}
			enc, err := h2m.MarshalBinary()
			if err != nil {
				t.Fatalf("MarshalBinary: %v", err)
			}
			if !bytes.Equal(enc, tt.golden) {
				t.Errorf("MarshalBinary = %x, want %x", enc, tt.golden)
			}
			h3u, ok := h3.(encoding.BinaryUnmarshaler)
			if !ok {
				t.Fatalf("Hash does not implement UnmarshalBinary")
			}
			if err := h3u.UnmarshalBinary(enc); err != nil {
				t.Fatalf("UnmarshalBinary: %v", err)
			}
			h2.Write(buf[split:])
			h3.Write(buf[split:])
			sum2 := h2.Sum(nil)
			sum3 := h3.Sum(nil)
			if !bytes.Equal(sum2, sum) {
				t.Fatalf("Sum after MarshalBinary = %x, want %x", sum2, sum)
			}
			if !bytes.Equal(sum3, sum) {
				t.Fatalf("Sum after UnmarshalBinary = %x, want %x", sum3, sum)
			}
		})
	}
}
