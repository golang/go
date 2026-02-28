// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cryptotest

import (
	"bytes"
	"crypto/cipher"
	"fmt"
	"testing"
)

var lengths = []int{0, 156, 8192, 8193, 8208}

// MakeAEAD returns a cipher.AEAD instance.
//
// Multiple calls to MakeAEAD must return equivalent instances, so for example
// the key must be fixed.
type MakeAEAD func() (cipher.AEAD, error)

// TestAEAD performs a set of tests on cipher.AEAD implementations, checking
// the documented requirements of NonceSize, Overhead, Seal and Open.
func TestAEAD(t *testing.T, mAEAD MakeAEAD) {
	aead, err := mAEAD()
	if err != nil {
		t.Fatal(err)
	}

	t.Run("Roundtrip", func(t *testing.T) {

		// Test all combinations of plaintext and additional data lengths.
		for _, ptLen := range lengths {
			for _, adLen := range lengths {
				t.Run(fmt.Sprintf("Plaintext-Length=%d,AddData-Length=%d", ptLen, adLen), func(t *testing.T) {
					rng := newRandReader(t)

					nonce := make([]byte, aead.NonceSize())
					rng.Read(nonce)

					before, addData := make([]byte, adLen), make([]byte, ptLen)
					rng.Read(before)
					rng.Read(addData)

					ciphertext := sealMsg(t, aead, nil, nonce, before, addData)
					after := openWithoutError(t, aead, nil, nonce, ciphertext, addData)

					if !bytes.Equal(after, before) {
						t.Errorf("plaintext is different after a seal/open cycle; got %s, want %s", truncateHex(after), truncateHex(before))
					}
				})
			}
		}
	})

	t.Run("InputNotModified", func(t *testing.T) {

		// Test all combinations of plaintext and additional data lengths.
		for _, ptLen := range lengths {
			for _, adLen := range lengths {
				t.Run(fmt.Sprintf("Plaintext-Length=%d,AddData-Length=%d", ptLen, adLen), func(t *testing.T) {
					t.Run("Seal", func(t *testing.T) {
						rng := newRandReader(t)

						nonce := make([]byte, aead.NonceSize())
						rng.Read(nonce)

						src, before := make([]byte, ptLen), make([]byte, ptLen)
						rng.Read(src)
						copy(before, src)

						addData := make([]byte, adLen)
						rng.Read(addData)

						sealMsg(t, aead, nil, nonce, src, addData)
						if !bytes.Equal(src, before) {
							t.Errorf("Seal modified src; got %s, want %s", truncateHex(src), truncateHex(before))
						}
					})

					t.Run("Open", func(t *testing.T) {
						rng := newRandReader(t)

						nonce := make([]byte, aead.NonceSize())
						rng.Read(nonce)

						plaintext, addData := make([]byte, ptLen), make([]byte, adLen)
						rng.Read(plaintext)
						rng.Read(addData)

						// Record the ciphertext that shouldn't be modified as the input of
						// Open.
						ciphertext := sealMsg(t, aead, nil, nonce, plaintext, addData)
						before := make([]byte, len(ciphertext))
						copy(before, ciphertext)

						openWithoutError(t, aead, nil, nonce, ciphertext, addData)
						if !bytes.Equal(ciphertext, before) {
							t.Errorf("Open modified src; got %s, want %s", truncateHex(ciphertext), truncateHex(before))
						}
					})
				})
			}
		}
	})

	t.Run("BufferOverlap", func(t *testing.T) {

		// Test all combinations of plaintext and additional data lengths.
		for _, ptLen := range lengths {
			if ptLen <= 1 { // We need enough room for an inexact overlap to occur.
				continue
			}
			for _, adLen := range lengths {
				t.Run(fmt.Sprintf("Plaintext-Length=%d,AddData-Length=%d", ptLen, adLen), func(t *testing.T) {
					t.Run("Seal", func(t *testing.T) {
						rng := newRandReader(t)

						nonce := make([]byte, aead.NonceSize())
						rng.Read(nonce)

						// Make a buffer that can hold a plaintext and ciphertext as we
						// overlap their slices to check for panic on inexact overlaps.
						ctLen := ptLen + aead.Overhead()
						buff := make([]byte, ptLen+ctLen)
						rng.Read(buff)

						addData := make([]byte, adLen)
						rng.Read(addData)

						// Make plaintext and dst slices point to same array with inexact overlap.
						plaintext := buff[:ptLen]
						dst := buff[1:1] // Shift dst to not start at start of plaintext.
						mustPanic(t, "invalid buffer overlap", func() { sealMsg(t, aead, dst, nonce, plaintext, addData) })

						// Only overlap on one byte
						plaintext = buff[:ptLen]
						dst = buff[ptLen-1 : ptLen-1]
						mustPanic(t, "invalid buffer overlap", func() { sealMsg(t, aead, dst, nonce, plaintext, addData) })
					})

					t.Run("Open", func(t *testing.T) {
						rng := newRandReader(t)

						nonce := make([]byte, aead.NonceSize())
						rng.Read(nonce)

						// Create a valid ciphertext to test Open with.
						plaintext := make([]byte, ptLen)
						rng.Read(plaintext)
						addData := make([]byte, adLen)
						rng.Read(addData)
						validCT := sealMsg(t, aead, nil, nonce, plaintext, addData)

						// Make a buffer that can hold a plaintext and ciphertext as we
						// overlap their slices to check for panic on inexact overlaps.
						buff := make([]byte, ptLen+len(validCT))

						// Make ciphertext and dst slices point to same array with inexact overlap.
						ciphertext := buff[:len(validCT)]
						copy(ciphertext, validCT)
						dst := buff[1:1] // Shift dst to not start at start of ciphertext.
						mustPanic(t, "invalid buffer overlap", func() { aead.Open(dst, nonce, ciphertext, addData) })

						// Only overlap on one byte.
						ciphertext = buff[:len(validCT)]
						copy(ciphertext, validCT)
						// Make sure it is the actual ciphertext being overlapped and not
						// the hash digest which might be extracted/truncated in some
						// implementations: Go one byte past the hash digest/tag and into
						// the ciphertext.
						beforeTag := len(validCT) - aead.Overhead()
						dst = buff[beforeTag-1 : beforeTag-1]
						mustPanic(t, "invalid buffer overlap", func() { aead.Open(dst, nonce, ciphertext, addData) })
					})
				})
			}
		}
	})

	t.Run("AppendDst", func(t *testing.T) {

		// Test all combinations of plaintext and additional data lengths.
		for _, ptLen := range lengths {
			for _, adLen := range lengths {
				t.Run(fmt.Sprintf("Plaintext-Length=%d,AddData-Length=%d", ptLen, adLen), func(t *testing.T) {

					t.Run("Seal", func(t *testing.T) {
						rng := newRandReader(t)

						nonce := make([]byte, aead.NonceSize())
						rng.Read(nonce)

						shortBuff := []byte("a")
						longBuff := make([]byte, 512)
						rng.Read(longBuff)
						prefixes := [][]byte{shortBuff, longBuff}

						// Check each prefix gets appended to by Seal without altering them.
						for _, prefix := range prefixes {
							plaintext, addData := make([]byte, ptLen), make([]byte, adLen)
							rng.Read(plaintext)
							rng.Read(addData)
							out := sealMsg(t, aead, prefix, nonce, plaintext, addData)

							// Check that Seal didn't alter the prefix
							if !bytes.Equal(out[:len(prefix)], prefix) {
								t.Errorf("Seal alters dst instead of appending; got %s, want %s", truncateHex(out[:len(prefix)]), truncateHex(prefix))
							}

							if isDeterministic(aead) {
								ciphertext := out[len(prefix):]
								// Check that the appended ciphertext wasn't affected by the prefix
								if expectedCT := sealMsg(t, aead, nil, nonce, plaintext, addData); !bytes.Equal(ciphertext, expectedCT) {
									t.Errorf("Seal behavior affected by pre-existing data in dst; got %s, want %s", truncateHex(ciphertext), truncateHex(expectedCT))
								}
							}
						}
					})

					t.Run("Open", func(t *testing.T) {
						rng := newRandReader(t)

						nonce := make([]byte, aead.NonceSize())
						rng.Read(nonce)

						shortBuff := []byte("a")
						longBuff := make([]byte, 512)
						rng.Read(longBuff)
						prefixes := [][]byte{shortBuff, longBuff}

						// Check each prefix gets appended to by Open without altering them.
						for _, prefix := range prefixes {
							before, addData := make([]byte, adLen), make([]byte, ptLen)
							rng.Read(before)
							rng.Read(addData)
							ciphertext := sealMsg(t, aead, nil, nonce, before, addData)

							out := openWithoutError(t, aead, prefix, nonce, ciphertext, addData)

							// Check that Open didn't alter the prefix
							if !bytes.Equal(out[:len(prefix)], prefix) {
								t.Errorf("Open alters dst instead of appending; got %s, want %s", truncateHex(out[:len(prefix)]), truncateHex(prefix))
							}

							after := out[len(prefix):]
							// Check that the appended plaintext wasn't affected by the prefix
							if !bytes.Equal(after, before) {
								t.Errorf("Open behavior affected by pre-existing data in dst; got %s, want %s", truncateHex(after), truncateHex(before))
							}
						}
					})
				})
			}
		}
	})

	t.Run("WrongNonce", func(t *testing.T) {
		if aead.NonceSize() == 0 {
			t.Skip("AEAD does not use a nonce")
		}
		// Test all combinations of plaintext and additional data lengths.
		for _, ptLen := range lengths {
			for _, adLen := range lengths {
				t.Run(fmt.Sprintf("Plaintext-Length=%d,AddData-Length=%d", ptLen, adLen), func(t *testing.T) {
					rng := newRandReader(t)

					nonce := make([]byte, aead.NonceSize())
					rng.Read(nonce)

					plaintext, addData := make([]byte, ptLen), make([]byte, adLen)
					rng.Read(plaintext)
					rng.Read(addData)

					ciphertext := sealMsg(t, aead, nil, nonce, plaintext, addData)

					// Perturb the nonce and check for an error when Opening
					alterNonce := make([]byte, aead.NonceSize())
					copy(alterNonce, nonce)
					alterNonce[len(alterNonce)-1] += 1
					_, err := aead.Open(nil, alterNonce, ciphertext, addData)

					if err == nil {
						t.Errorf("Open did not error when given different nonce than Sealed with")
					}
				})
			}
		}
	})

	t.Run("WrongAddData", func(t *testing.T) {

		// Test all combinations of plaintext and additional data lengths.
		for _, ptLen := range lengths {
			for _, adLen := range lengths {
				if adLen == 0 {
					continue
				}

				t.Run(fmt.Sprintf("Plaintext-Length=%d,AddData-Length=%d", ptLen, adLen), func(t *testing.T) {
					rng := newRandReader(t)

					nonce := make([]byte, aead.NonceSize())
					rng.Read(nonce)

					plaintext, addData := make([]byte, ptLen), make([]byte, adLen)
					rng.Read(plaintext)
					rng.Read(addData)

					ciphertext := sealMsg(t, aead, nil, nonce, plaintext, addData)

					// Perturb the Additional Data and check for an error when Opening
					alterAD := make([]byte, adLen)
					copy(alterAD, addData)
					alterAD[len(alterAD)-1] += 1
					_, err := aead.Open(nil, nonce, ciphertext, alterAD)

					if err == nil {
						t.Errorf("Open did not error when given different Additional Data than Sealed with")
					}
				})
			}
		}
	})

	t.Run("WrongCiphertext", func(t *testing.T) {

		// Test all combinations of plaintext and additional data lengths.
		for _, ptLen := range lengths {
			for _, adLen := range lengths {

				t.Run(fmt.Sprintf("Plaintext-Length=%d,AddData-Length=%d", ptLen, adLen), func(t *testing.T) {
					rng := newRandReader(t)

					nonce := make([]byte, aead.NonceSize())
					rng.Read(nonce)

					plaintext, addData := make([]byte, ptLen), make([]byte, adLen)
					rng.Read(plaintext)
					rng.Read(addData)

					ciphertext := sealMsg(t, aead, nil, nonce, plaintext, addData)

					// Perturb the ciphertext and check for an error when Opening
					alterCT := make([]byte, len(ciphertext))
					copy(alterCT, ciphertext)
					alterCT[len(alterCT)-1] += 1
					_, err := aead.Open(nil, nonce, alterCT, addData)

					if err == nil {
						t.Errorf("Open did not error when given different ciphertext than was produced by Seal")
					}
				})
			}
		}
	})
}

// Helper function to Seal a plaintext with additional data. Checks that
// ciphertext isn't bigger than the plaintext length plus Overhead()
func sealMsg(t *testing.T, aead cipher.AEAD, ciphertext, nonce, plaintext, addData []byte) []byte {
	t.Helper()

	initialLen := len(ciphertext)

	ciphertext = aead.Seal(ciphertext, nonce, plaintext, addData)

	lenCT := len(ciphertext) - initialLen

	// Appended ciphertext shouldn't ever be longer than the length of the
	// plaintext plus Overhead
	if lenCT > len(plaintext)+aead.Overhead() {
		t.Errorf("length of ciphertext from Seal exceeds length of plaintext by more than Overhead(); got %d, want <=%d", lenCT, len(plaintext)+aead.Overhead())
	}

	return ciphertext
}

func isDeterministic(aead cipher.AEAD) bool {
	// Check if the AEAD is deterministic by checking if the same plaintext
	// encrypted with the same nonce and additional data produces the same
	// ciphertext.
	nonce := make([]byte, aead.NonceSize())
	addData := []byte("additional data")
	plaintext := []byte("plaintext")
	ciphertext1 := aead.Seal(nil, nonce, plaintext, addData)
	ciphertext2 := aead.Seal(nil, nonce, plaintext, addData)
	return bytes.Equal(ciphertext1, ciphertext2)
}

// Helper function to Open and authenticate ciphertext. Checks that Open
// doesn't error (assuming ciphertext was well-formed with corresponding nonce
// and additional data).
func openWithoutError(t *testing.T, aead cipher.AEAD, plaintext, nonce, ciphertext, addData []byte) []byte {
	t.Helper()

	plaintext, err := aead.Open(plaintext, nonce, ciphertext, addData)
	if err != nil {
		t.Fatalf("Open returned error on properly formed ciphertext; got \"%s\", want \"nil\"", err)
	}

	return plaintext
}
