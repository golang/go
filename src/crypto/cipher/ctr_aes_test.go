// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// CTR AES test vectors.

// See U.S. National Institute of Standards and Technology (NIST)
// Special Publication 800-38A, ``Recommendation for Block Cipher
// Modes of Operation,'' 2001 Edition, pp. 55-58.

package cipher_test

import (
	"bytes"
	"crypto/aes"
	"crypto/cipher"
	"crypto/internal/boring"
	"crypto/internal/cryptotest"
	fipsaes "crypto/internal/fips/aes"
	"encoding/hex"
	"fmt"
	"math/rand"
	"sort"
	"strings"
	"testing"
)

var commonCounter = []byte{0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa, 0xfb, 0xfc, 0xfd, 0xfe, 0xff}

var ctrAESTests = []struct {
	name string
	key  []byte
	iv   []byte
	in   []byte
	out  []byte
}{
	// NIST SP 800-38A pp 55-58
	{
		"CTR-AES128",
		commonKey128,
		commonCounter,
		commonInput,
		[]byte{
			0x87, 0x4d, 0x61, 0x91, 0xb6, 0x20, 0xe3, 0x26, 0x1b, 0xef, 0x68, 0x64, 0x99, 0x0d, 0xb6, 0xce,
			0x98, 0x06, 0xf6, 0x6b, 0x79, 0x70, 0xfd, 0xff, 0x86, 0x17, 0x18, 0x7b, 0xb9, 0xff, 0xfd, 0xff,
			0x5a, 0xe4, 0xdf, 0x3e, 0xdb, 0xd5, 0xd3, 0x5e, 0x5b, 0x4f, 0x09, 0x02, 0x0d, 0xb0, 0x3e, 0xab,
			0x1e, 0x03, 0x1d, 0xda, 0x2f, 0xbe, 0x03, 0xd1, 0x79, 0x21, 0x70, 0xa0, 0xf3, 0x00, 0x9c, 0xee,
		},
	},
	{
		"CTR-AES192",
		commonKey192,
		commonCounter,
		commonInput,
		[]byte{
			0x1a, 0xbc, 0x93, 0x24, 0x17, 0x52, 0x1c, 0xa2, 0x4f, 0x2b, 0x04, 0x59, 0xfe, 0x7e, 0x6e, 0x0b,
			0x09, 0x03, 0x39, 0xec, 0x0a, 0xa6, 0xfa, 0xef, 0xd5, 0xcc, 0xc2, 0xc6, 0xf4, 0xce, 0x8e, 0x94,
			0x1e, 0x36, 0xb2, 0x6b, 0xd1, 0xeb, 0xc6, 0x70, 0xd1, 0xbd, 0x1d, 0x66, 0x56, 0x20, 0xab, 0xf7,
			0x4f, 0x78, 0xa7, 0xf6, 0xd2, 0x98, 0x09, 0x58, 0x5a, 0x97, 0xda, 0xec, 0x58, 0xc6, 0xb0, 0x50,
		},
	},
	{
		"CTR-AES256",
		commonKey256,
		commonCounter,
		commonInput,
		[]byte{
			0x60, 0x1e, 0xc3, 0x13, 0x77, 0x57, 0x89, 0xa5, 0xb7, 0xa7, 0xf5, 0x04, 0xbb, 0xf3, 0xd2, 0x28,
			0xf4, 0x43, 0xe3, 0xca, 0x4d, 0x62, 0xb5, 0x9a, 0xca, 0x84, 0xe9, 0x90, 0xca, 0xca, 0xf5, 0xc5,
			0x2b, 0x09, 0x30, 0xda, 0xa2, 0x3d, 0xe9, 0x4c, 0xe8, 0x70, 0x17, 0xba, 0x2d, 0x84, 0x98, 0x8d,
			0xdf, 0xc9, 0xc5, 0x8d, 0xb6, 0x7a, 0xad, 0xa6, 0x13, 0xc2, 0xdd, 0x08, 0x45, 0x79, 0x41, 0xa6,
		},
	},
}

func TestCTR_AES(t *testing.T) {
	cryptotest.TestAllImplementations(t, "aes", testCTR_AES)
}

func testCTR_AES(t *testing.T) {
	for _, tt := range ctrAESTests {
		test := tt.name

		c, err := aes.NewCipher(tt.key)
		if err != nil {
			t.Errorf("%s: NewCipher(%d bytes) = %s", test, len(tt.key), err)
			continue
		}

		for j := 0; j <= 5; j += 5 {
			in := tt.in[0 : len(tt.in)-j]
			ctr := cipher.NewCTR(c, tt.iv)
			encrypted := make([]byte, len(in))
			ctr.XORKeyStream(encrypted, in)
			if out := tt.out[:len(in)]; !bytes.Equal(out, encrypted) {
				t.Errorf("%s/%d: CTR\ninpt %x\nhave %x\nwant %x", test, len(in), in, encrypted, out)
			}
		}

		for j := 0; j <= 7; j += 7 {
			in := tt.out[0 : len(tt.out)-j]
			ctr := cipher.NewCTR(c, tt.iv)
			plain := make([]byte, len(in))
			ctr.XORKeyStream(plain, in)
			if out := tt.in[:len(in)]; !bytes.Equal(out, plain) {
				t.Errorf("%s/%d: CTRReader\nhave %x\nwant %x", test, len(out), plain, out)
			}
		}

		if t.Failed() {
			break
		}
	}
}

func makeTestingCiphers(aesBlock cipher.Block, iv []byte) (genericCtr, multiblockCtr cipher.Stream) {
	return cipher.NewCTR(wrap(aesBlock), iv), cipher.NewCTR(aesBlock, iv)
}

func randBytes(t *testing.T, r *rand.Rand, count int) []byte {
	t.Helper()
	buf := make([]byte, count)
	n, err := r.Read(buf)
	if err != nil {
		t.Fatal(err)
	}
	if n != count {
		t.Fatal("short read from Rand")
	}
	return buf
}

const aesBlockSize = 16

type ctrAble interface {
	NewCTR(iv []byte) cipher.Stream
}

// Verify that multiblock AES CTR (src/crypto/aes/ctr_*.s)
// produces the same results as generic single-block implementation.
// This test runs checks on random IV.
func TestCTR_AES_multiblock_random_IV(t *testing.T) {
	r := rand.New(rand.NewSource(54321))
	iv := randBytes(t, r, aesBlockSize)
	const Size = 100

	for _, keySize := range []int{16, 24, 32} {
		keySize := keySize
		t.Run(fmt.Sprintf("keySize=%d", keySize), func(t *testing.T) {
			key := randBytes(t, r, keySize)
			aesBlock, err := aes.NewCipher(key)
			if err != nil {
				t.Fatal(err)
			}
			genericCtr, _ := makeTestingCiphers(aesBlock, iv)

			plaintext := randBytes(t, r, Size)

			// Generate reference ciphertext.
			genericCiphertext := make([]byte, len(plaintext))
			genericCtr.XORKeyStream(genericCiphertext, plaintext)

			// Split the text in 3 parts in all possible ways and encrypt them
			// individually using multiblock implementation to catch edge cases.

			for part1 := 0; part1 <= Size; part1++ {
				part1 := part1
				t.Run(fmt.Sprintf("part1=%d", part1), func(t *testing.T) {
					for part2 := 0; part2 <= Size-part1; part2++ {
						part2 := part2
						t.Run(fmt.Sprintf("part2=%d", part2), func(t *testing.T) {
							_, multiblockCtr := makeTestingCiphers(aesBlock, iv)
							multiblockCiphertext := make([]byte, len(plaintext))
							multiblockCtr.XORKeyStream(multiblockCiphertext[:part1], plaintext[:part1])
							multiblockCtr.XORKeyStream(multiblockCiphertext[part1:part1+part2], plaintext[part1:part1+part2])
							multiblockCtr.XORKeyStream(multiblockCiphertext[part1+part2:], plaintext[part1+part2:])
							if !bytes.Equal(genericCiphertext, multiblockCiphertext) {
								t.Fatal("multiblock CTR's output does not match generic CTR's output")
							}
						})
					}
				})
			}
		})
	}
}

func parseHex(str string) []byte {
	b, err := hex.DecodeString(strings.ReplaceAll(str, " ", ""))
	if err != nil {
		panic(err)
	}
	return b
}

// Verify that multiblock AES CTR (src/crypto/aes/ctr_*.s)
// produces the same results as generic single-block implementation.
// This test runs checks on edge cases (IV overflows).
func TestCTR_AES_multiblock_overflow_IV(t *testing.T) {
	r := rand.New(rand.NewSource(987654))

	const Size = 4096
	plaintext := randBytes(t, r, Size)

	ivs := [][]byte{
		parseHex("00 00 00 00 00 00 00 00   FF FF FF FF FF FF FF FF"),
		parseHex("FF FF FF FF FF FF FF FF   FF FF FF FF FF FF FF FF"),
		parseHex("FF FF FF FF FF FF FF FF   00 00 00 00 00 00 00 00"),
		parseHex("FF FF FF FF FF FF FF FF   FF FF FF FF FF FF FF fe"),
		parseHex("00 00 00 00 00 00 00 00   FF FF FF FF FF FF FF fe"),
		parseHex("FF FF FF FF FF FF FF FF   FF FF FF FF FF FF FF 00"),
		parseHex("00 00 00 00 00 00 00 01   FF FF FF FF FF FF FF 00"),
		parseHex("00 00 00 00 00 00 00 01   FF FF FF FF FF FF FF FF"),
		parseHex("00 00 00 00 00 00 00 01   FF FF FF FF FF FF FF fe"),
		parseHex("00 00 00 00 00 00 00 01   FF FF FF FF FF FF FF 00"),
	}

	for _, keySize := range []int{16, 24, 32} {
		keySize := keySize
		t.Run(fmt.Sprintf("keySize=%d", keySize), func(t *testing.T) {
			for _, iv := range ivs {
				key := randBytes(t, r, keySize)
				aesBlock, err := aes.NewCipher(key)
				if err != nil {
					t.Fatal(err)
				}

				t.Run(fmt.Sprintf("iv=%s", hex.EncodeToString(iv)), func(t *testing.T) {
					for _, offset := range []int{0, 1, 16, 1024} {
						offset := offset
						t.Run(fmt.Sprintf("offset=%d", offset), func(t *testing.T) {
							genericCtr, multiblockCtr := makeTestingCiphers(aesBlock, iv)

							// Generate reference ciphertext.
							genericCiphertext := make([]byte, Size)
							genericCtr.XORKeyStream(genericCiphertext, plaintext)

							multiblockCiphertext := make([]byte, Size)
							multiblockCtr.XORKeyStream(multiblockCiphertext, plaintext[:offset])
							multiblockCtr.XORKeyStream(multiblockCiphertext[offset:], plaintext[offset:])
							if !bytes.Equal(genericCiphertext, multiblockCiphertext) {
								t.Fatal("multiblock CTR's output does not match generic CTR's output")
							}
						})
					}
				})
			}
		})
	}
}

// Check that method XORKeyStreamAt works correctly.
func TestCTR_AES_multiblock_XORKeyStreamAt(t *testing.T) {
	if boring.Enabled {
		t.Skip("XORKeyStreamAt is not available in boring mode")
	}

	r := rand.New(rand.NewSource(12345))
	const Size = 32 * 1024 * 1024
	plaintext := randBytes(t, r, Size)

	for _, keySize := range []int{16, 24, 32} {
		keySize := keySize
		t.Run(fmt.Sprintf("keySize=%d", keySize), func(t *testing.T) {
			key := randBytes(t, r, keySize)
			iv := randBytes(t, r, aesBlockSize)

			aesBlock, err := aes.NewCipher(key)
			if err != nil {
				t.Fatal(err)
			}
			genericCtr, _ := makeTestingCiphers(aesBlock, iv)
			ctrAt := fipsaes.NewCTR(aesBlock.(*fipsaes.Block), iv)

			// Generate reference ciphertext.
			genericCiphertext := make([]byte, Size)
			genericCtr.XORKeyStream(genericCiphertext, plaintext)

			multiblockCiphertext := make([]byte, Size)
			// Split the range to random slices.
			const N = 1000
			boundaries := make([]int, 0, N+2)
			for i := 0; i < N; i++ {
				boundaries = append(boundaries, r.Intn(Size))
			}
			boundaries = append(boundaries, 0)
			boundaries = append(boundaries, Size)
			sort.Ints(boundaries)

			for _, i := range r.Perm(N + 1) {
				begin := boundaries[i]
				end := boundaries[i+1]
				ctrAt.XORKeyStreamAt(
					multiblockCiphertext[begin:end],
					plaintext[begin:end],
					uint64(begin),
				)
			}

			if !bytes.Equal(genericCiphertext, multiblockCiphertext) {
				t.Fatal("multiblock CTR's output does not match generic CTR's output")
			}
		})
	}
}
