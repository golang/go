// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"encoding/hex"
	"testing"
)

type testSplitPreMasterSecretTest struct {
	in, out1, out2 string
}

var testSplitPreMasterSecretTests = []testSplitPreMasterSecretTest{
	{"", "", ""},
	{"00", "00", "00"},
	{"0011", "00", "11"},
	{"001122", "0011", "1122"},
	{"00112233", "0011", "2233"},
}

func TestSplitPreMasterSecret(t *testing.T) {
	for i, test := range testSplitPreMasterSecretTests {
		in, _ := hex.DecodeString(test.in)
		out1, out2 := splitPreMasterSecret(in)
		s1 := hex.EncodeToString(out1)
		s2 := hex.EncodeToString(out2)
		if s1 != test.out1 || s2 != test.out2 {
			t.Errorf("#%d: got: (%s, %s) want: (%s, %s)", i, s1, s2, test.out1, test.out2)
		}
	}
}

type testKeysFromTest struct {
	version                                        uint16
	suite                                          *cipherSuite
	preMasterSecret                                string
	clientRandom, serverRandom                     string
	masterSecret                                   string
	clientMAC, serverMAC                           string
	clientKey, serverKey                           string
	macLen, keyLen                                 int
	contextKeyingMaterial, noContextKeyingMaterial string
}

func TestKeysFromPreMasterSecret(t *testing.T) {
	for i, test := range testKeysFromTests {
		in, _ := hex.DecodeString(test.preMasterSecret)
		clientRandom, _ := hex.DecodeString(test.clientRandom)
		serverRandom, _ := hex.DecodeString(test.serverRandom)

		masterSecret := masterFromPreMasterSecret(test.version, test.suite, in, clientRandom, serverRandom)
		if s := hex.EncodeToString(masterSecret); s != test.masterSecret {
			t.Errorf("#%d: bad master secret %s, want %s", i, s, test.masterSecret)
			continue
		}

		clientMAC, serverMAC, clientKey, serverKey, _, _ := keysFromMasterSecret(test.version, test.suite, masterSecret, clientRandom, serverRandom, test.macLen, test.keyLen, 0)
		clientMACString := hex.EncodeToString(clientMAC)
		serverMACString := hex.EncodeToString(serverMAC)
		clientKeyString := hex.EncodeToString(clientKey)
		serverKeyString := hex.EncodeToString(serverKey)
		if clientMACString != test.clientMAC ||
			serverMACString != test.serverMAC ||
			clientKeyString != test.clientKey ||
			serverKeyString != test.serverKey {
			t.Errorf("#%d: got: (%s, %s, %s, %s) want: (%s, %s, %s, %s)", i, clientMACString, serverMACString, clientKeyString, serverKeyString, test.clientMAC, test.serverMAC, test.clientKey, test.serverKey)
		}

		ekm := ekmFromMasterSecret(test.version, test.suite, masterSecret, clientRandom, serverRandom)
		contextKeyingMaterial, err := ekm("label", []byte("context"), 32)
		if err != nil {
			t.Fatalf("ekmFromMasterSecret failed: %v", err)
		}

		noContextKeyingMaterial, err := ekm("label", nil, 32)
		if err != nil {
			t.Fatalf("ekmFromMasterSecret failed: %v", err)
		}

		if hex.EncodeToString(contextKeyingMaterial) != test.contextKeyingMaterial ||
			hex.EncodeToString(noContextKeyingMaterial) != test.noContextKeyingMaterial {
			t.Errorf("#%d: got keying material: (%s, %s) want: (%s, %s)", i, contextKeyingMaterial, noContextKeyingMaterial, test.contextKeyingMaterial, test.noContextKeyingMaterial)
		}
	}
}

// These test vectors were generated from GnuTLS using `gnutls-cli --insecure -d 9 `
var testKeysFromTests = []testKeysFromTest{
	{
		VersionTLS10,
		cipherSuiteByID(TLS_RSA_WITH_RC4_128_SHA),
		"0302cac83ad4b1db3b9ab49ad05957de2a504a634a386fc600889321e1a971f57479466830ac3e6f468e87f5385fa0c5",
		"4ae66303755184a3917fcb44880605fcc53baa01912b22ed94473fc69cebd558",
		"4ae663020ec16e6bb5130be918cfcafd4d765979a3136a5d50c593446e4e44db",
		"3d851bab6e5556e959a16bc36d66cfae32f672bfa9ecdef6096cbb1b23472df1da63dbbd9827606413221d149ed08ceb",
		"805aaa19b3d2c0a0759a4b6c9959890e08480119",
		"2d22f9fe519c075c16448305ceee209fc24ad109",
		"d50b5771244f850cd8117a9ccafe2cf1",
		"e076e33206b30507a85c32855acd0919",
		20,
		16,
		"4d1bb6fc278c37d27aa6e2a13c2e079095d143272c2aa939da33d88c1c0cec22",
		"93fba89599b6321ae538e27c6548ceb8b46821864318f5190d64a375e5d69d41",
	},
	{
		VersionTLS10,
		cipherSuiteByID(TLS_RSA_WITH_RC4_128_SHA),
		"03023f7527316bc12cbcd69e4b9e8275d62c028f27e65c745cfcddc7ce01bd3570a111378b63848127f1c36e5f9e4890",
		"4ae66364b5ea56b20ce4e25555aed2d7e67f42788dd03f3fee4adae0459ab106",
		"4ae66363ab815cbf6a248b87d6b556184e945e9b97fbdf247858b0bdafacfa1c",
		"7d64be7c80c59b740200b4b9c26d0baaa1c5ae56705acbcf2307fe62beb4728c19392c83f20483801cce022c77645460",
		"97742ed60a0554ca13f04f97ee193177b971e3b0",
		"37068751700400e03a8477a5c7eec0813ab9e0dc",
		"207cddbc600d2a200abac6502053ee5c",
		"df3f94f6e1eacc753b815fe16055cd43",
		20,
		16,
		"2c9f8961a72b97cbe76553b5f954caf8294fc6360ef995ac1256fe9516d0ce7f",
		"274f19c10291d188857ad8878e2119f5aa437d4da556601cf1337aff23154016",
	},
	{
		VersionTLS10,
		cipherSuiteByID(TLS_RSA_WITH_RC4_128_SHA),
		"832d515f1d61eebb2be56ba0ef79879efb9b527504abb386fb4310ed5d0e3b1f220d3bb6b455033a2773e6d8bdf951d278a187482b400d45deb88a5d5a6bb7d6a7a1decc04eb9ef0642876cd4a82d374d3b6ff35f0351dc5d411104de431375355addc39bfb1f6329fb163b0bc298d658338930d07d313cd980a7e3d9196cac1",
		"4ae663b2ee389c0de147c509d8f18f5052afc4aaf9699efe8cb05ece883d3a5e",
		"4ae664d503fd4cff50cfc1fb8fc606580f87b0fcdac9554ba0e01d785bdf278e",
		"1aff2e7a2c4279d0126f57a65a77a8d9d0087cf2733366699bec27eb53d5740705a8574bb1acc2abbe90e44f0dd28d6c",
		"3c7647c93c1379a31a609542aa44e7f117a70085",
		"0d73102994be74a575a3ead8532590ca32a526d4",
		"ac7581b0b6c10d85bbd905ffbf36c65e",
		"ff07edde49682b45466bd2e39464b306",
		20,
		16,
		"678b0d43f607de35241dc7e9d1a7388a52c35033a1a0336d4d740060a6638fe2",
		"f3b4ac743f015ef21d79978297a53da3e579ee047133f38c234d829c0f907dab",
	},
}
