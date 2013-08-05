// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cipher_test

import (
	"bytes"
	"crypto/aes"
	"crypto/cipher"
	"encoding/hex"
	"testing"
)

// AES-GCM test vectors taken from gcmEncryptExtIV128.rsp from
// http://csrc.nist.gov/groups/STM/cavp/index.html.
var aesGCMTests = []struct {
	key, nonce, plaintext, ad, result string
}{
	{
		"11754cd72aec309bf52f7687212e8957",
		"3c819d9a9bed087615030b65",
		"",
		"",
		"250327c674aaf477aef2675748cf6971",
	},
	{
		"ca47248ac0b6f8372a97ac43508308ed",
		"ffd2b598feabc9019262d2be",
		"",
		"",
		"60d20404af527d248d893ae495707d1a",
	},
	{
		"77be63708971c4e240d1cb79e8d77feb",
		"e0e00f19fed7ba0136a797f3",
		"",
		"7a43ec1d9c0a5a78a0b16533a6213cab",
		"209fcc8d3675ed938e9c7166709dd946",
	},
	{
		"7680c5d3ca6154758e510f4d25b98820",
		"f8f105f9c3df4965780321f8",
		"",
		"c94c410194c765e3dcc7964379758ed3",
		"94dca8edfcf90bb74b153c8d48a17930",
	},
	{
		"7fddb57453c241d03efbed3ac44e371c",
		"ee283a3fc75575e33efd4887",
		"d5de42b461646c255c87bd2962d3b9a2",
		"",
		"2ccda4a5415cb91e135c2a0f78c9b2fdb36d1df9b9d5e596f83e8b7f52971cb3",
	},
	{
		"ab72c77b97cb5fe9a382d9fe81ffdbed",
		"54cc7dc2c37ec006bcc6d1da",
		"007c5e5b3e59df24a7c355584fc1518d",
		"",
		"0e1bde206a07a9c2c1b65300f8c649972b4401346697138c7a4891ee59867d0c",
	},
	{
		"fe47fcce5fc32665d2ae399e4eec72ba",
		"5adb9609dbaeb58cbd6e7275",
		"7c0e88c88899a779228465074797cd4c2e1498d259b54390b85e3eef1c02df60e743f1b840382c4bccaf3bafb4ca8429bea063",
		"88319d6e1d3ffa5f987199166c8a9b56c2aeba5a",
		"98f4826f05a265e6dd2be82db241c0fbbbf9ffb1c173aa83964b7cf5393043736365253ddbc5db8778371495da76d269e5db3e291ef1982e4defedaa2249f898556b47",
	},
	{
		"ec0c2ba17aa95cd6afffe949da9cc3a8",
		"296bce5b50b7d66096d627ef",
		"b85b3753535b825cbe5f632c0b843c741351f18aa484281aebec2f45bb9eea2d79d987b764b9611f6c0f8641843d5d58f3a242",
		"f8d00f05d22bf68599bcdeb131292ad6e2df5d14",
		"a7443d31c26bdf2a1c945e29ee4bd344a99cfaf3aa71f8b3f191f83c2adfc7a07162995506fde6309ffc19e716eddf1a828c5a890147971946b627c40016da1ecf3e77",
	},
	{
		"2c1f21cf0f6fb3661943155c3e3d8492",
		"23cb5ff362e22426984d1907",
		"42f758836986954db44bf37c6ef5e4ac0adaf38f27252a1b82d02ea949c8a1a2dbc0d68b5615ba7c1220ff6510e259f06655d8",
		"5d3624879d35e46849953e45a32a624d6a6c536ed9857c613b572b0333e701557a713e3f010ecdf9a6bd6c9e3e44b065208645aff4aabee611b391528514170084ccf587177f4488f33cfb5e979e42b6e1cfc0a60238982a7aec",
		"81824f0e0d523db30d3da369fdc0d60894c7a0a20646dd015073ad2732bd989b14a222b6ad57af43e1895df9dca2a5344a62cc57a3ee28136e94c74838997ae9823f3a",
	},
	{
		"d9f7d2411091f947b4d6f1e2d1f0fb2e",
		"e1934f5db57cc983e6b180e7",
		"73ed042327f70fe9c572a61545eda8b2a0c6e1d6c291ef19248e973aee6c312012f490c2c6f6166f4a59431e182663fcaea05a",
		"0a8a18a7150e940c3d87b38e73baee9a5c049ee21795663e264b694a949822b639092d0e67015e86363583fcf0ca645af9f43375f05fdb4ce84f411dcbca73c2220dea03a20115d2e51398344b16bee1ed7c499b353d6c597af8",
		"aaadbd5c92e9151ce3db7210b8714126b73e43436d242677afa50384f2149b831f1d573c7891c2a91fbc48db29967ec9542b2321b51ca862cb637cdd03b99a0f93b134",
	},
	{
		"fe9bb47deb3a61e423c2231841cfd1fb",
		"4d328eb776f500a2f7fb47aa",
		"f1cc3818e421876bb6b8bbd6c9",
		"",
		"b88c5c1977b35b517b0aeae96743fd4727fe5cdb4b5b42818dea7ef8c9",
	},
	{
		"6703df3701a7f54911ca72e24dca046a",
		"12823ab601c350ea4bc2488c",
		"793cd125b0b84a043e3ac67717",
		"",
		"b2051c80014f42f08735a7b0cd38e6bcd29962e5f2c13626b85a877101",
	},
}

func TestAESGCM(t *testing.T) {
	for i, test := range aesGCMTests {
		key, _ := hex.DecodeString(test.key)
		aes, err := aes.NewCipher(key)
		if err != nil {
			t.Fatal(err)
		}

		nonce, _ := hex.DecodeString(test.nonce)
		plaintext, _ := hex.DecodeString(test.plaintext)
		ad, _ := hex.DecodeString(test.ad)
		aesgcm, err := cipher.NewGCM(aes)
		if err != nil {
			t.Fatal(err)
		}

		ct := aesgcm.Seal(nil, nonce, plaintext, ad)
		if ctHex := hex.EncodeToString(ct); ctHex != test.result {
			t.Errorf("#%d: got %s, want %s", i, ctHex, test.result)
			continue
		}

		plaintext2, err := aesgcm.Open(nil, nonce, ct, ad)
		if err != nil {
			t.Errorf("#%d: Open failed", i)
			continue
		}

		if !bytes.Equal(plaintext, plaintext2) {
			t.Errorf("#%d: plaintext's don't match: got %x vs %x", i, plaintext2, plaintext)
			continue
		}

		if len(ad) > 0 {
			ad[0] ^= 0x80
			if _, err := aesgcm.Open(nil, nonce, ct, ad); err == nil {
				t.Errorf("#%d: Open was successful after altering additional data", i)
			}
			ad[0] ^= 0x80
		}

		nonce[0] ^= 0x80
		if _, err := aesgcm.Open(nil, nonce, ct, ad); err == nil {
			t.Errorf("#%d: Open was successful after altering nonce", i)
		}
		nonce[0] ^= 0x80

		ct[0] ^= 0x80
		if _, err := aesgcm.Open(nil, nonce, ct, ad); err == nil {
			t.Errorf("#%d: Open was successful after altering ciphertext", i)
		}
		ct[0] ^= 0x80
	}
}

func BenchmarkAESGCM(b *testing.B) {
	buf := make([]byte, 1024)
	b.SetBytes(int64(len(buf)))

	var key [16]byte
	var nonce [12]byte
	aes, _ := aes.NewCipher(key[:])
	aesgcm, _ := cipher.NewGCM(aes)
	var out []byte

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out = aesgcm.Seal(out[:0], nonce[:], buf, nonce[:])
	}
}
