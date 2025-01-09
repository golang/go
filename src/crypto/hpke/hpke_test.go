// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hpke

import (
	"bytes"
	"crypto/ecdh"
	"crypto/mlkem"
	"crypto/mlkem/mlkemtest"
	"crypto/sha3"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"testing"
)

func Example() {
	// In this example, we use MLKEM768-X25519 as the KEM, HKDF-SHA256 as the
	// KDF, and AES-256-GCM as the AEAD to encrypt a single message from a
	// sender to a recipient using the one-shot API.

	kem, kdf, aead := MLKEM768X25519(), HKDFSHA256(), AES256GCM()

	// Recipient side
	var (
		recipientPrivateKey PrivateKey
		publicKeyBytes      []byte
	)
	{
		k, err := kem.GenerateKey()
		if err != nil {
			panic(err)
		}
		recipientPrivateKey = k
		publicKeyBytes = k.PublicKey().Bytes()
	}

	// Sender side
	var ciphertext []byte
	{
		publicKey, err := kem.NewPublicKey(publicKeyBytes)
		if err != nil {
			panic(err)
		}

		message := []byte("|-()-|")
		ct, err := Seal(publicKey, kdf, aead, []byte("example"), message)
		if err != nil {
			panic(err)
		}

		ciphertext = ct
	}

	// Recipient side
	{
		plaintext, err := Open(recipientPrivateKey, kdf, aead, []byte("example"), ciphertext)
		if err != nil {
			panic(err)
		}
		fmt.Printf("Decrypted message: %s\n", plaintext)
	}

	// Output:
	// Decrypted message: |-()-|
}

func TestRoundTrip(t *testing.T) {
	kems := []KEM{
		DHKEM(ecdh.P256()),
		DHKEM(ecdh.P384()),
		DHKEM(ecdh.P521()),
		DHKEM(ecdh.X25519()),
		MLKEM768(),
		MLKEM1024(),
		MLKEM768P256(),
		MLKEM1024P384(),
		MLKEM768X25519(),
	}
	kdfs := []KDF{
		HKDFSHA256(),
		HKDFSHA384(),
		HKDFSHA512(),
		SHAKE128(),
		SHAKE256(),
	}
	aeads := []AEAD{
		AES128GCM(),
		AES256GCM(),
		ChaCha20Poly1305(),
	}

	for _, kem := range kems {
		t.Run(fmt.Sprintf("KEM_%04x", kem.ID()), func(t *testing.T) {
			k, err := kem.GenerateKey()
			if err != nil {
				t.Fatal(err)
			}
			kb, err := k.Bytes()
			if err != nil {
				t.Fatal(err)
			}
			kk, err := kem.NewPrivateKey(kb)
			if err != nil {
				t.Fatal(err)
			}
			if got, err := kk.Bytes(); err != nil {
				t.Fatal(err)
			} else if !bytes.Equal(got, kb) {
				t.Errorf("re-serialized key mismatch: got %x, want %x", got, kb)
			}
			pk, err := kem.NewPublicKey(k.PublicKey().Bytes())
			if err != nil {
				t.Fatal(err)
			}
			if got := pk.Bytes(); !bytes.Equal(got, k.PublicKey().Bytes()) {
				t.Errorf("re-serialized public key mismatch: got %x, want %x", got, k.PublicKey().Bytes())
			}

			for _, kdf := range kdfs {
				t.Run(fmt.Sprintf("KDF_%04x", kdf.ID()), func(t *testing.T) {
					for _, aead := range aeads {
						t.Run(fmt.Sprintf("AEAD_%04x", aead.ID()), func(t *testing.T) {
							c, err := Seal(pk, kdf, aead, []byte("info"), []byte("plaintext"))
							if err != nil {
								t.Fatal(err)
							}
							p, err := Open(kk, kdf, aead, []byte("info"), c)
							if err != nil {
								t.Fatal(err)
							}
							if !bytes.Equal(p, []byte("plaintext")) {
								t.Errorf("unexpected plaintext: got %x, want %x", p, []byte("plaintext"))
							}

							p, err = Open(kk, kdf, aead, []byte("wrong"), c)
							if err == nil {
								t.Errorf("expected error when opening with wrong info, got plaintext %x", p)
							}
							c[len(c)-1] ^= 0xFF
							p, err = Open(kk, kdf, aead, []byte("info"), c)
							if err == nil {
								t.Errorf("expected error when opening with corrupted ciphertext, got plaintext %x", p)
							}

							c, err = Seal(k.PublicKey(), kdf, aead, nil, nil)
							if err != nil {
								t.Fatal(err)
							}
							p, err = Open(k, kdf, aead, nil, c)
							if err != nil {
								t.Fatal(err)
							}
							if len(p) != 0 {
								t.Errorf("unexpected plaintext: got %x, want empty", p)
							}

							// Test that Seal and Open don't modify the excess capacity of input
							// slices. This is a regression test for a bug where decap would
							// append to the enc slice, corrupting the ciphertext if they shared
							// a backing array.
							padSlice := func(b []byte) []byte {
								s := make([]byte, len(b), len(b)+2000)
								copy(s, b)
								for i := len(b); i < cap(s); i++ {
									s[:cap(s)][i] = 0xAA
								}
								return s[:len(b)]
							}
							checkSlice := func(name string, s []byte) {
								for i := len(s); i < cap(s); i++ {
									if s[:cap(s)][i] != 0xAA {
										t.Errorf("%s: modified byte at index %d beyond slice length", name, i)
										return
									}
								}
							}

							infoS := padSlice([]byte("info"))
							plaintextS := padSlice([]byte("plaintext"))
							c, err = Seal(pk, kdf, aead, infoS, plaintextS)
							if err != nil {
								t.Fatal(err)
							}
							checkSlice("Seal info", infoS)
							checkSlice("Seal plaintext", plaintextS)

							infoO := padSlice([]byte("info"))
							ciphertextO := padSlice(c)
							p, err = Open(kk, kdf, aead, infoO, ciphertextO)
							if err != nil {
								t.Fatalf("Open with large capacity slices failed: %v", err)
							}
							if !bytes.Equal(p, []byte("plaintext")) {
								t.Errorf("unexpected plaintext: got %x, want %x", p, []byte("plaintext"))
							}
							checkSlice("Open info", infoO)
							checkSlice("Open ciphertext", ciphertextO)

							// Also test the Sender.Seal and Recipient.Open methods.
							infoSender := padSlice([]byte("info"))
							enc, sender, err := NewSender(pk, kdf, aead, infoSender)
							if err != nil {
								t.Fatal(err)
							}
							checkSlice("NewSender info", infoSender)

							aadSeal := padSlice([]byte("aad"))
							plaintextSeal := padSlice([]byte("plaintext"))
							ct, err := sender.Seal(aadSeal, plaintextSeal)
							if err != nil {
								t.Fatal(err)
							}
							checkSlice("Sender.Seal aad", aadSeal)
							checkSlice("Sender.Seal plaintext", plaintextSeal)

							infoRecipient := padSlice([]byte("info"))
							encPadded := padSlice(enc)
							recipient, err := NewRecipient(encPadded, kk, kdf, aead, infoRecipient)
							if err != nil {
								t.Fatal(err)
							}
							checkSlice("NewRecipient info", infoRecipient)
							checkSlice("NewRecipient enc", encPadded)

							aadOpen := padSlice([]byte("aad"))
							ctPadded := padSlice(ct)
							p, err = recipient.Open(aadOpen, ctPadded)
							if err != nil {
								t.Fatalf("Recipient.Open failed: %v", err)
							}
							if !bytes.Equal(p, []byte("plaintext")) {
								t.Errorf("unexpected plaintext: got %x, want %x", p, []byte("plaintext"))
							}
							checkSlice("Recipient.Open aad", aadOpen)
							checkSlice("Recipient.Open ciphertext", ctPadded)
						})
					}
				})
			}
		})
	}
}

func mustDecodeHex(t *testing.T, in string) []byte {
	t.Helper()
	b, err := hex.DecodeString(in)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestVectors(t *testing.T) {
	t.Run("rfc9180", func(t *testing.T) {
		testVectors(t, "rfc9180")
	})
	t.Run("hpke-pq", func(t *testing.T) {
		testVectors(t, "hpke-pq")
	})
}

func testVectors(t *testing.T, name string) {
	vectorsJSON, err := os.ReadFile("testdata/" + name + ".json")
	if err != nil {
		t.Fatal(err)
	}
	var vectors []struct {
		Mode        uint16 `json:"mode"`
		KEM         uint16 `json:"kem_id"`
		KDF         uint16 `json:"kdf_id"`
		AEAD        uint16 `json:"aead_id"`
		Info        string `json:"info"`
		IkmE        string `json:"ikmE"`
		IkmR        string `json:"ikmR"`
		SkRm        string `json:"skRm"`
		PkRm        string `json:"pkRm"`
		Enc         string `json:"enc"`
		Encryptions []struct {
			Aad   string `json:"aad"`
			Ct    string `json:"ct"`
			Nonce string `json:"nonce"`
			Pt    string `json:"pt"`
		} `json:"encryptions"`
		Exports []struct {
			Context string `json:"exporter_context"`
			L       int    `json:"L"`
			Value   string `json:"exported_value"`
		} `json:"exports"`

		// Instead of checking in a very large rfc9180.json, we computed
		// alternative accumulated values.
		AccEncryptions string `json:"encryptions_accumulated"`
		AccExports     string `json:"exports_accumulated"`
	}
	if err := json.Unmarshal(vectorsJSON, &vectors); err != nil {
		t.Fatal(err)
	}

	for _, vector := range vectors {
		name := fmt.Sprintf("mode %04x kem %04x kdf %04x aead %04x",
			vector.Mode, vector.KEM, vector.KDF, vector.AEAD)
		t.Run(name, func(t *testing.T) {
			if vector.Mode != 0 {
				t.Skip("only mode 0 (base) is supported")
			}
			if vector.KEM == 0x0021 {
				t.Skip("KEM 0x0021 (DHKEM(X448)) not supported")
			}
			if vector.KEM == 0x0040 {
				t.Skip("KEM 0x0040 (ML-KEM-512) not supported")
			}
			if vector.KDF == 0x0012 || vector.KDF == 0x0013 {
				t.Skipf("TurboSHAKE KDF not supported")
			}

			kdf, err := NewKDF(vector.KDF)
			if err != nil {
				t.Fatal(err)
			}
			if kdf.ID() != vector.KDF {
				t.Errorf("unexpected KDF ID: got %04x, want %04x", kdf.ID(), vector.KDF)
			}

			aead, err := NewAEAD(vector.AEAD)
			if err != nil {
				t.Fatal(err)
			}
			if aead.ID() != vector.AEAD {
				t.Errorf("unexpected AEAD ID: got %04x, want %04x", aead.ID(), vector.AEAD)
			}

			kem, err := NewKEM(vector.KEM)
			if err != nil {
				t.Fatal(err)
			}
			if kem.ID() != vector.KEM {
				t.Errorf("unexpected KEM ID: got %04x, want %04x", kem.ID(), vector.KEM)
			}

			pubKeyBytes := mustDecodeHex(t, vector.PkRm)
			kemSender, err := kem.NewPublicKey(pubKeyBytes)
			if err != nil {
				t.Fatal(err)
			}
			if kemSender.KEM() != kem {
				t.Errorf("unexpected KEM from sender: got %04x, want %04x", kemSender.KEM().ID(), kem.ID())
			}
			if !bytes.Equal(kemSender.Bytes(), pubKeyBytes) {
				t.Errorf("unexpected KEM bytes: got %x, want %x", kemSender.Bytes(), pubKeyBytes)
			}

			ikmE := mustDecodeHex(t, vector.IkmE)
			setupDerandomizedEncap(t, ikmE, kemSender)

			info := mustDecodeHex(t, vector.Info)
			encap, sender, err := NewSender(kemSender, kdf, aead, info)
			if err != nil {
				t.Fatal(err)
			}
			if len(encap) != kem.encSize() {
				t.Errorf("unexpected encapsulated key size: got %d, want %d", len(encap), kem.encSize())
			}

			expectedEncap := mustDecodeHex(t, vector.Enc)
			if !bytes.Equal(encap, expectedEncap) {
				t.Errorf("unexpected encapsulated key, got: %x, want %x", encap, expectedEncap)
			}

			privKeyBytes := mustDecodeHex(t, vector.SkRm)
			kemRecipient, err := kem.NewPrivateKey(privKeyBytes)
			if err != nil {
				t.Fatal(err)
			}
			if kemRecipient.KEM() != kem {
				t.Errorf("unexpected KEM from recipient: got %04x, want %04x", kemRecipient.KEM().ID(), kem.ID())
			}
			kemRecipientBytes, err := kemRecipient.Bytes()
			if err != nil {
				t.Fatal(err)
			}
			// X25519 serialized keys must be clamped, so the bytes might not match.
			if !bytes.Equal(kemRecipientBytes, privKeyBytes) && vector.KEM != DHKEM(ecdh.X25519()).ID() {
				t.Errorf("unexpected KEM bytes: got %x, want %x", kemRecipientBytes, privKeyBytes)
			}
			if vector.KEM == DHKEM(ecdh.X25519()).ID() {
				kem2, err := kem.NewPrivateKey(kemRecipientBytes)
				if err != nil {
					t.Fatal(err)
				}
				kemRecipientBytes2, err := kem2.Bytes()
				if err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(kemRecipientBytes2, kemRecipientBytes) {
					t.Errorf("X25519 re-serialized key differs: got %x, want %x", kemRecipientBytes2, kemRecipientBytes)
				}
				if !bytes.Equal(kem2.PublicKey().Bytes(), pubKeyBytes) {
					t.Errorf("X25519 re-derived public key differs: got %x, want %x", kem2.PublicKey().Bytes(), pubKeyBytes)
				}
			}
			if !bytes.Equal(kemRecipient.PublicKey().Bytes(), pubKeyBytes) {
				t.Errorf("unexpected KEM sender bytes: got %x, want %x", kemRecipient.PublicKey().Bytes(), pubKeyBytes)
			}

			ikm := mustDecodeHex(t, vector.IkmR)
			derivRecipient, err := kem.DeriveKeyPair(ikm)
			if err != nil {
				t.Fatal(err)
			}
			derivRecipientBytes, err := derivRecipient.Bytes()
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(derivRecipientBytes, privKeyBytes) && vector.KEM != DHKEM(ecdh.X25519()).ID() {
				t.Errorf("unexpected KEM bytes from seed: got %x, want %x", derivRecipientBytes, privKeyBytes)
			}
			if !bytes.Equal(derivRecipient.PublicKey().Bytes(), pubKeyBytes) {
				t.Errorf("unexpected KEM sender bytes from seed: got %x, want %x", derivRecipient.PublicKey().Bytes(), pubKeyBytes)
			}

			recipient, err := NewRecipient(encap, kemRecipient, kdf, aead, info)
			if err != nil {
				t.Fatal(err)
			}

			if aead != ExportOnly() && len(vector.AccEncryptions) != 0 {
				source, sink := sha3.NewSHAKE128(), sha3.NewSHAKE128()
				for range 1000 {
					aad, plaintext := drawRandomInput(t, source), drawRandomInput(t, source)
					ciphertext, err := sender.Seal(aad, plaintext)
					if err != nil {
						t.Fatal(err)
					}
					sink.Write(ciphertext)
					got, err := recipient.Open(aad, ciphertext)
					if err != nil {
						t.Fatal(err)
					}
					if !bytes.Equal(got, plaintext) {
						t.Errorf("unexpected plaintext: got %x want %x", got, plaintext)
					}
				}
				encryptions := make([]byte, 16)
				sink.Read(encryptions)
				expectedEncryptions := mustDecodeHex(t, vector.AccEncryptions)
				if !bytes.Equal(encryptions, expectedEncryptions) {
					t.Errorf("unexpected accumulated encryptions, got: %x, want %x", encryptions, expectedEncryptions)
				}
			} else if aead != ExportOnly() {
				for _, enc := range vector.Encryptions {
					aad := mustDecodeHex(t, enc.Aad)
					plaintext := mustDecodeHex(t, enc.Pt)
					expectedCiphertext := mustDecodeHex(t, enc.Ct)

					ciphertext, err := sender.Seal(aad, plaintext)
					if err != nil {
						t.Fatal(err)
					}
					if !bytes.Equal(ciphertext, expectedCiphertext) {
						t.Errorf("unexpected ciphertext, got: %x, want %x", ciphertext, expectedCiphertext)
					}

					got, err := recipient.Open(aad, ciphertext)
					if err != nil {
						t.Fatal(err)
					}
					if !bytes.Equal(got, plaintext) {
						t.Errorf("unexpected plaintext: got %x want %x", got, plaintext)
					}
				}
			} else {
				if _, err := sender.Seal(nil, nil); err == nil {
					t.Error("expected error from Seal with export-only AEAD")
				}
				if _, err := recipient.Open(nil, nil); err == nil {
					t.Error("expected error from Open with export-only AEAD")
				}
			}

			if len(vector.AccExports) != 0 {
				source, sink := sha3.NewSHAKE128(), sha3.NewSHAKE128()
				for l := range 1000 {
					context := string(drawRandomInput(t, source))
					value, err := sender.Export(context, l)
					if err != nil {
						t.Fatal(err)
					}
					sink.Write(value)
					got, err := recipient.Export(context, l)
					if err != nil {
						t.Fatal(err)
					}
					if !bytes.Equal(got, value) {
						t.Errorf("recipient: unexpected exported secret: got %x want %x", got, value)
					}
				}
				exports := make([]byte, 16)
				sink.Read(exports)
				expectedExports := mustDecodeHex(t, vector.AccExports)
				if !bytes.Equal(exports, expectedExports) {
					t.Errorf("unexpected accumulated exports, got: %x, want %x", exports, expectedExports)
				}
			} else {
				for _, exp := range vector.Exports {
					context := string(mustDecodeHex(t, exp.Context))
					expectedValue := mustDecodeHex(t, exp.Value)

					value, err := sender.Export(context, exp.L)
					if err != nil {
						t.Fatal(err)
					}
					if !bytes.Equal(value, expectedValue) {
						t.Errorf("unexpected exported value, got: %x, want %x", value, expectedValue)
					}

					got, err := recipient.Export(context, exp.L)
					if err != nil {
						t.Fatal(err)
					}
					if !bytes.Equal(got, value) {
						t.Errorf("recipient: unexpected exported secret: got %x want %x", got, value)
					}
				}
			}
		})
	}
}

func drawRandomInput(t *testing.T, r io.Reader) []byte {
	t.Helper()
	l := make([]byte, 1)
	if _, err := r.Read(l); err != nil {
		t.Fatal(err)
	}
	n := int(l[0])
	b := make([]byte, n)
	if _, err := r.Read(b); err != nil {
		t.Fatal(err)
	}
	return b
}

func setupDerandomizedEncap(t *testing.T, randBytes []byte, pk PublicKey) {
	t.Cleanup(func() {
		testingOnlyGenerateKey = nil
		testingOnlyEncapsulate = nil
	})
	switch pk.KEM() {
	case DHKEM(ecdh.P256()), DHKEM(ecdh.P384()), DHKEM(ecdh.P521()), DHKEM(ecdh.X25519()):
		r, err := pk.KEM().DeriveKeyPair(randBytes)
		if err != nil {
			t.Fatal(err)
		}
		testingOnlyGenerateKey = func() *ecdh.PrivateKey {
			return r.(*dhKEMPrivateKey).priv.(*ecdh.PrivateKey)
		}
	case mlkem768:
		pq := pk.(*mlkemPublicKey).pq.(*mlkem.EncapsulationKey768)
		testingOnlyEncapsulate = func() ([]byte, []byte) {
			ss, ct, err := mlkemtest.Encapsulate768(pq, randBytes)
			if err != nil {
				t.Fatal(err)
			}
			return ss, ct
		}
	case mlkem1024:
		pq := pk.(*mlkemPublicKey).pq.(*mlkem.EncapsulationKey1024)
		testingOnlyEncapsulate = func() ([]byte, []byte) {
			ss, ct, err := mlkemtest.Encapsulate1024(pq, randBytes)
			if err != nil {
				t.Fatal(err)
			}
			return ss, ct
		}
	case mlkem768X25519:
		pqRand, tRand := randBytes[:32], randBytes[32:]
		pq := pk.(*hybridPublicKey).pq.(*mlkem.EncapsulationKey768)
		k, err := ecdh.X25519().NewPrivateKey(tRand)
		if err != nil {
			t.Fatal(err)
		}
		testingOnlyGenerateKey = func() *ecdh.PrivateKey {
			return k
		}
		testingOnlyEncapsulate = func() ([]byte, []byte) {
			ss, ct, err := mlkemtest.Encapsulate768(pq, pqRand)
			if err != nil {
				t.Fatal(err)
			}
			return ss, ct
		}
	case mlkem768P256:
		// The rest of randBytes are the following candidates for rejection
		// sampling, but they are never reached.
		pqRand, tRand := randBytes[:32], randBytes[32:64]
		pq := pk.(*hybridPublicKey).pq.(*mlkem.EncapsulationKey768)
		k, err := ecdh.P256().NewPrivateKey(tRand)
		if err != nil {
			t.Fatal(err)
		}
		testingOnlyGenerateKey = func() *ecdh.PrivateKey {
			return k
		}
		testingOnlyEncapsulate = func() ([]byte, []byte) {
			ss, ct, err := mlkemtest.Encapsulate768(pq, pqRand)
			if err != nil {
				t.Fatal(err)
			}
			return ss, ct
		}
	case mlkem1024P384:
		pqRand, tRand := randBytes[:32], randBytes[32:]
		pq := pk.(*hybridPublicKey).pq.(*mlkem.EncapsulationKey1024)
		k, err := ecdh.P384().NewPrivateKey(tRand)
		if err != nil {
			t.Fatal(err)
		}
		testingOnlyGenerateKey = func() *ecdh.PrivateKey {
			return k
		}
		testingOnlyEncapsulate = func() ([]byte, []byte) {
			ss, ct, err := mlkemtest.Encapsulate1024(pq, pqRand)
			if err != nil {
				t.Fatal(err)
			}
			return ss, ct
		}
	default:
		t.Fatalf("unsupported KEM %04x", pk.KEM().ID())
	}
}

func TestSingletons(t *testing.T) {
	if HKDFSHA256() != HKDFSHA256() {
		t.Error("HKDFSHA256() != HKDFSHA256()")
	}
	if HKDFSHA384() != HKDFSHA384() {
		t.Error("HKDFSHA384() != HKDFSHA384()")
	}
	if HKDFSHA512() != HKDFSHA512() {
		t.Error("HKDFSHA512() != HKDFSHA512()")
	}
	if AES128GCM() != AES128GCM() {
		t.Error("AES128GCM() != AES128GCM()")
	}
	if AES256GCM() != AES256GCM() {
		t.Error("AES256GCM() != AES256GCM()")
	}
	if ChaCha20Poly1305() != ChaCha20Poly1305() {
		t.Error("ChaCha20Poly1305() != ChaCha20Poly1305()")
	}
	if ExportOnly() != ExportOnly() {
		t.Error("ExportOnly() != ExportOnly()")
	}
	if DHKEM(ecdh.P256()) != DHKEM(ecdh.P256()) {
		t.Error("DHKEM(P-256) != DHKEM(P-256)")
	}
	if DHKEM(ecdh.P384()) != DHKEM(ecdh.P384()) {
		t.Error("DHKEM(P-384) != DHKEM(P-384)")
	}
	if DHKEM(ecdh.P521()) != DHKEM(ecdh.P521()) {
		t.Error("DHKEM(P-521) != DHKEM(P-521)")
	}
	if DHKEM(ecdh.X25519()) != DHKEM(ecdh.X25519()) {
		t.Error("DHKEM(X25519) != DHKEM(X25519)")
	}
	if MLKEM768() != MLKEM768() {
		t.Error("MLKEM768() != MLKEM768()")
	}
	if MLKEM1024() != MLKEM1024() {
		t.Error("MLKEM1024() != MLKEM1024()")
	}
	if MLKEM768X25519() != MLKEM768X25519() {
		t.Error("MLKEM768X25519() != MLKEM768X25519()")
	}
	if MLKEM768P256() != MLKEM768P256() {
		t.Error("MLKEM768P256() != MLKEM768P256()")
	}
	if MLKEM1024P384() != MLKEM1024P384() {
		t.Error("MLKEM1024P384() != MLKEM1024P384()")
	}
}
