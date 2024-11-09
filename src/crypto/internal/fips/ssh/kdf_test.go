// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh_test

import (
	"bytes"
	"crypto/internal/fips/ssh"
	"crypto/sha256"
	"encoding/hex"
	"testing"
)

func TestACVPVector(t *testing.T) {
	// https://github.com/usnistgov/ACVP-Server/blob/3a7333f638/gen-val/json-files/kdf-components-ssh-1.0/prompt.json#L910-L915
	K := fromHex("0000010100E534CD9780786AF19994DD68C3FD7FE1E1F77C3938B2005C49B080CF88A63A44079774A36F23BA4D73470CB318C30524854D2F36BAB9A45AD73DBB3BC5DD39A547F62BC921052E102E37F3DD0CD79A04EB46ACC14B823B326096A89E33E8846624188BB3C8F16B320E7BB8F5EB05F080DCEE244A445DBED3A9F3BA8C373D8BE62CDFE2FC5876F30F90F01F0A55E5251B23E0DBBFCFB1450715E329BB00FB222E850DDB11201460B8AEF3FC8965D3B6D3AFBB885A6C11F308F10211B82EA2028C7A84DD0BB8D5D6AC3A48D0C2B93609269C585E03889DB3621993E7F7C09A007FB6B5C06FFA532B0DBF11F71F740D9CD8FAD2532E21B9423BF3D85EE4E396BE32")
	H := fromHex("8FB22F0864960DA5679FD377248E41C2D0390E5AB3BB7955A3B6C588FB75B20D")
	sessionID := fromHex("269A512E7B560E13396E0F3F56BDA730E23EE122EE6D59C91C58FB07872BCCCC")

	// https://github.com/usnistgov/ACVP-Server/blob/3a7333f638/gen-val/json-files/kdf-components-ssh-1.0/expectedResults.json#L1306-L1314
	initialIVClient := fromHex("82321D9FE2ACD958D3F55F4D3FF5C79D")
	initialIVServer := fromHex("03F336F61311770BD5346B41E04CDB1F")
	encryptionKeyClient := fromHex("20E55008D0120C400F42E5D2E148AB75")
	encryptionKeyServer := fromHex("8BF4DEBEC96F4ADBBE5BB43828D56E6D")
	integrityKeyClient := fromHex("15F53BCCE2645D0AD1C539C09BF9054AA3A4B10B71E96B9E3A15672405341BB5")
	integrityKeyServer := fromHex("00BB773FD63AC7B7281A7B54C130CCAD363EE8928104E67CA5A3211EE3BBAB93")

	gotIVClient, gotKeyClient, gotIntegrityClient := ssh.Keys(
		sha256.New, ssh.ClientKeys, K, H, sessionID, 16, 16, 32)
	gotIVServer, gotKeyServer, gotIntegrityServer := ssh.Keys(
		sha256.New, ssh.ServerKeys, K, H, sessionID, 16, 16, 32)

	if !bytes.Equal(gotIVClient, initialIVClient) {
		t.Errorf("got IV client %x, want %x", gotIVClient, initialIVClient)
	}
	if !bytes.Equal(gotKeyClient, encryptionKeyClient) {
		t.Errorf("got key client %x, want %x", gotKeyClient, encryptionKeyClient)
	}
	if !bytes.Equal(gotIntegrityClient, integrityKeyClient) {
		t.Errorf("got integrity key client %x, want %x", gotIntegrityClient, integrityKeyClient)
	}
	if !bytes.Equal(gotIVServer, initialIVServer) {
		t.Errorf("got IV server %x, want %x", gotIVServer, initialIVServer)
	}
	if !bytes.Equal(gotKeyServer, encryptionKeyServer) {
		t.Errorf("got key server %x, want %x", gotKeyServer, encryptionKeyServer)
	}
	if !bytes.Equal(gotIntegrityServer, integrityKeyServer) {
		t.Errorf("got integrity key server %x, want %x", gotIntegrityServer, integrityKeyServer)
	}
}

func fromHex(s string) []byte {
	b, err := hex.DecodeString(s)
	if err != nil {
		panic(err)
	}
	return b
}
