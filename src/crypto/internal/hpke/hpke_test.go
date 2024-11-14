// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hpke

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"os"
	"strconv"
	"strings"
	"testing"

	"crypto/ecdh"
	_ "crypto/sha256"
	_ "crypto/sha512"
)

func mustDecodeHex(t *testing.T, in string) []byte {
	b, err := hex.DecodeString(in)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func parseVectorSetup(vector string) map[string]string {
	vals := map[string]string{}
	for _, l := range strings.Split(vector, "\n") {
		fields := strings.Split(l, ": ")
		vals[fields[0]] = fields[1]
	}
	return vals
}

func parseVectorEncryptions(vector string) []map[string]string {
	vals := []map[string]string{}
	for _, section := range strings.Split(vector, "\n\n") {
		e := map[string]string{}
		for _, l := range strings.Split(section, "\n") {
			fields := strings.Split(l, ": ")
			e[fields[0]] = fields[1]
		}
		vals = append(vals, e)
	}
	return vals
}

func TestRFC9180Vectors(t *testing.T) {
	vectorsJSON, err := os.ReadFile("testdata/rfc9180-vectors.json")
	if err != nil {
		t.Fatal(err)
	}

	var vectors []struct {
		Name        string
		Setup       string
		Encryptions string
	}
	if err := json.Unmarshal(vectorsJSON, &vectors); err != nil {
		t.Fatal(err)
	}

	for _, vector := range vectors {
		t.Run(vector.Name, func { t ->
			setup := parseVectorSetup(vector.Setup)

			kemID, err := strconv.Atoi(setup["kem_id"])
			if err != nil {
				t.Fatal(err)
			}
			if _, ok := SupportedKEMs[uint16(kemID)]; !ok {
				t.Skip("unsupported KEM")
			}
			kdfID, err := strconv.Atoi(setup["kdf_id"])
			if err != nil {
				t.Fatal(err)
			}
			if _, ok := SupportedKDFs[uint16(kdfID)]; !ok {
				t.Skip("unsupported KDF")
			}
			aeadID, err := strconv.Atoi(setup["aead_id"])
			if err != nil {
				t.Fatal(err)
			}
			if _, ok := SupportedAEADs[uint16(aeadID)]; !ok {
				t.Skip("unsupported AEAD")
			}

			info := mustDecodeHex(t, setup["info"])
			pubKeyBytes := mustDecodeHex(t, setup["pkRm"])
			pub, err := ParseHPKEPublicKey(uint16(kemID), pubKeyBytes)
			if err != nil {
				t.Fatal(err)
			}

			ephemeralPrivKey := mustDecodeHex(t, setup["skEm"])

			testingOnlyGenerateKey = func { SupportedKEMs[uint16(kemID)].curve.NewPrivateKey(ephemeralPrivKey) }
			t.Cleanup(func() { testingOnlyGenerateKey = nil })

			encap, context, err := SetupSender(
				uint16(kemID),
				uint16(kdfID),
				uint16(aeadID),
				pub,
				info,
			)
			if err != nil {
				t.Fatal(err)
			}

			expectedEncap := mustDecodeHex(t, setup["enc"])
			if !bytes.Equal(encap, expectedEncap) {
				t.Errorf("unexpected encapsulated key, got: %x, want %x", encap, expectedEncap)
			}
			expectedSharedSecret := mustDecodeHex(t, setup["shared_secret"])
			if !bytes.Equal(context.sharedSecret, expectedSharedSecret) {
				t.Errorf("unexpected shared secret, got: %x, want %x", context.sharedSecret, expectedSharedSecret)
			}
			expectedKey := mustDecodeHex(t, setup["key"])
			if !bytes.Equal(context.key, expectedKey) {
				t.Errorf("unexpected key, got: %x, want %x", context.key, expectedKey)
			}
			expectedBaseNonce := mustDecodeHex(t, setup["base_nonce"])
			if !bytes.Equal(context.baseNonce, expectedBaseNonce) {
				t.Errorf("unexpected base nonce, got: %x, want %x", context.baseNonce, expectedBaseNonce)
			}
			expectedExporterSecret := mustDecodeHex(t, setup["exporter_secret"])
			if !bytes.Equal(context.exporterSecret, expectedExporterSecret) {
				t.Errorf("unexpected exporter secret, got: %x, want %x", context.exporterSecret, expectedExporterSecret)
			}

			for _, enc := range parseVectorEncryptions(vector.Encryptions) {
				t.Run("seq num "+enc["sequence number"], func { t ->
					seqNum, err := strconv.Atoi(enc["sequence number"])
					if err != nil {
						t.Fatal(err)
					}
					context.seqNum = uint128{lo: uint64(seqNum)}
					expectedNonce := mustDecodeHex(t, enc["nonce"])
					// We can't call nextNonce, because it increments the sequence number,
					// so just compute it directly.
					computedNonce := context.seqNum.bytes()[16-context.aead.NonceSize():]
					for i := range context.baseNonce {
						computedNonce[i] ^= context.baseNonce[i]
					}
					if !bytes.Equal(computedNonce, expectedNonce) {
						t.Errorf("unexpected nonce: got %x, want %x", computedNonce, expectedNonce)
					}

					expectedCiphertext := mustDecodeHex(t, enc["ct"])
					ciphertext, err := context.Seal(mustDecodeHex(t, enc["aad"]), mustDecodeHex(t, enc["pt"]))
					if err != nil {
						t.Fatal(err)
					}
					if !bytes.Equal(ciphertext, expectedCiphertext) {
						t.Errorf("unexpected ciphertext: got %x want %x", ciphertext, expectedCiphertext)
					}
				})
			}
		})
	}
}
