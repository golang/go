// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packet

import (
	"bytes"
	"crypto"
	"encoding/hex"
	"testing"
)

func TestSignatureRead(t *testing.T) {
	packet, err := Read(readerFromHex(signatureDataHex))
	if err != nil {
		t.Error(err)
		return
	}
	sig, ok := packet.(*Signature)
	if !ok || sig.SigType != SigTypeBinary || sig.PubKeyAlgo != PubKeyAlgoRSA || sig.Hash != crypto.SHA1 {
		t.Errorf("failed to parse, got: %#v", packet)
	}
}

func TestSignatureReserialize(t *testing.T) {
	packet, _ := Read(readerFromHex(signatureDataHex))
	sig := packet.(*Signature)
	out := new(bytes.Buffer)
	err := sig.Serialize(out)
	if err != nil {
		t.Errorf("error reserializing: %s", err)
		return
	}

	expected, _ := hex.DecodeString(signatureDataHex)
	if !bytes.Equal(expected, out.Bytes()) {
		t.Errorf("output doesn't match input (got vs expected):\n%s\n%s", hex.Dump(out.Bytes()), hex.Dump(expected))
	}
}

const signatureDataHex = "c2c05c04000102000605024cb45112000a0910ab105c91af38fb158f8d07ff5596ea368c5efe015bed6e78348c0f033c931d5f2ce5db54ce7f2a7e4b4ad64db758d65a7a71773edeab7ba2a9e0908e6a94a1175edd86c1d843279f045b021a6971a72702fcbd650efc393c5474d5b59a15f96d2eaad4c4c426797e0dcca2803ef41c6ff234d403eec38f31d610c344c06f2401c262f0993b2e66cad8a81ebc4322c723e0d4ba09fe917e8777658307ad8329adacba821420741009dfe87f007759f0982275d028a392c6ed983a0d846f890b36148c7358bdb8a516007fac760261ecd06076813831a36d0459075d1befa245ae7f7fb103d92ca759e9498fe60ef8078a39a3beda510deea251ea9f0a7f0df6ef42060f20780360686f3e400e"
