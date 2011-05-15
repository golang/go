// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packet

import (
	"bytes"
	"encoding/hex"
	"testing"
)

var pubKeyTests = []struct {
	hexData        string
	hexFingerprint string
	creationTime   uint32
	pubKeyAlgo     PublicKeyAlgorithm
	keyId          uint64
	keyIdString    string
	keyIdShort     string
}{
	{rsaPkDataHex, rsaFingerprintHex, 0x4d3c5c10, PubKeyAlgoRSA, 0xa34d7e18c20c31bb, "A34D7E18C20C31BB", "C20C31BB"},
	{dsaPkDataHex, dsaFingerprintHex, 0x4d432f89, PubKeyAlgoDSA, 0x8e8fbe54062f19ed, "8E8FBE54062F19ED", "062F19ED"},
}

func TestPublicKeyRead(t *testing.T) {
	for i, test := range pubKeyTests {
		packet, err := Read(readerFromHex(test.hexData))
		if err != nil {
			t.Errorf("#%d: Read error: %s", i, err)
			continue
		}
		pk, ok := packet.(*PublicKey)
		if !ok {
			t.Errorf("#%d: failed to parse, got: %#v", i, packet)
			continue
		}
		if pk.PubKeyAlgo != test.pubKeyAlgo {
			t.Errorf("#%d: bad public key algorithm got:%x want:%x", i, pk.PubKeyAlgo, test.pubKeyAlgo)
		}
		if pk.CreationTime != test.creationTime {
			t.Errorf("#%d: bad creation time got:%x want:%x", i, pk.CreationTime, test.creationTime)
		}
		expectedFingerprint, _ := hex.DecodeString(test.hexFingerprint)
		if !bytes.Equal(expectedFingerprint, pk.Fingerprint[:]) {
			t.Errorf("#%d: bad fingerprint got:%x want:%x", i, pk.Fingerprint[:], expectedFingerprint)
		}
		if pk.KeyId != test.keyId {
			t.Errorf("#%d: bad keyid got:%x want:%x", i, pk.KeyId, test.keyId)
		}
		if g, e := pk.KeyIdString(), test.keyIdString; g != e {
			t.Errorf("#%d: bad KeyIdString got:%q want:%q", i, g, e)
		}
		if g, e := pk.KeyIdShortString(), test.keyIdShort; g != e {
			t.Errorf("#%d: bad KeyIdShortString got:%q want:%q", i, g, e)
		}
	}
}

func TestPublicKeySerialize(t *testing.T) {
	for i, test := range pubKeyTests {
		packet, err := Read(readerFromHex(test.hexData))
		if err != nil {
			t.Errorf("#%d: Read error: %s", i, err)
			continue
		}
		pk, ok := packet.(*PublicKey)
		if !ok {
			t.Errorf("#%d: failed to parse, got: %#v", i, packet)
			continue
		}
		serializeBuf := bytes.NewBuffer(nil)
		err = pk.Serialize(serializeBuf)
		if err != nil {
			t.Errorf("#%d: failed to serialize: %s", i, err)
			continue
		}

		packet, err = Read(serializeBuf)
		if err != nil {
			t.Errorf("#%d: Read error (from serialized data): %s", i, err)
			continue
		}
		pk, ok = packet.(*PublicKey)
		if !ok {
			t.Errorf("#%d: failed to parse serialized data, got: %#v", i, packet)
			continue
		}
	}
}

const rsaFingerprintHex = "5fb74b1d03b1e3cb31bc2f8aa34d7e18c20c31bb"

const rsaPkDataHex = "988d044d3c5c10010400b1d13382944bd5aba23a4312968b5095d14f947f600eb478e14a6fcb16b0e0cac764884909c020bc495cfcc39a935387c661507bdb236a0612fb582cac3af9b29cc2c8c70090616c41b662f4da4c1201e195472eb7f4ae1ccbcbf9940fe21d985e379a5563dde5b9a23d35f1cfaa5790da3b79db26f23695107bfaca8e7b5bcd0011010001"

const dsaFingerprintHex = "eece4c094db002103714c63c8e8fbe54062f19ed"

const dsaPkDataHex = "9901a2044d432f89110400cd581334f0d7a1e1bdc8b9d6d8c0baf68793632735d2bb0903224cbaa1dfbf35a60ee7a13b92643421e1eb41aa8d79bea19a115a677f6b8ba3c7818ce53a6c2a24a1608bd8b8d6e55c5090cbde09dd26e356267465ae25e69ec8bdd57c7bbb2623e4d73336f73a0a9098f7f16da2e25252130fd694c0e8070c55a812a423ae7f00a0ebf50e70c2f19c3520a551bd4b08d30f23530d3d03ff7d0bf4a53a64a09dc5e6e6e35854b7d70c882b0c60293401958b1bd9e40abec3ea05ba87cf64899299d4bd6aa7f459c201d3fbbd6c82004bdc5e8a9eb8082d12054cc90fa9d4ec251a843236a588bf49552441817436c4f43326966fe85447d4e6d0acf8fa1ef0f014730770603ad7634c3088dc52501c237328417c31c89ed70400b2f1a98b0bf42f11fefc430704bebbaa41d9f355600c3facee1e490f64208e0e094ea55e3a598a219a58500bf78ac677b670a14f4e47e9cf8eab4f368cc1ddcaa18cc59309d4cc62dd4f680e73e6cc3e1ce87a84d0925efbcb26c575c093fc42eecf45135fabf6403a25c2016e1774c0484e440a18319072c617cc97ac0a3bb0"
