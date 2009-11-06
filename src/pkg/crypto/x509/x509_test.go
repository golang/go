// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

import "testing"

func TestToKeepGoTestHappy(t *testing.T) {
}

/*

Div is broken for this key in 32-bit mode.

TODO(agl): reenabled when Div is fixed.

import (
	"big";
	"crypto/rsa";
	"encoding/pem";
	"reflect";
	"strings";
	"testing";
)

func TestParsePKCS1PrivateKey(t *testing.T) {
	block, _ := pem.Decode(strings.Bytes(pemPrivateKey));
	priv, err := ParsePKCS1PrivateKey(block.Bytes);
	if err != nil {
		t.Errorf("Failed to parse private key: %s", err);
	}
	if !reflect.DeepEqual(priv, rsaPrivateKey) {
		t.Errorf("got:%+v want:%+v", priv, rsaPrivateKey);
	}
}

var pemPrivateKey = `-----BEGIN RSA PRIVATE KEY-----
MIIBOgIBAAJBALKZD0nEffqM1ACuak0bijtqE2QrI/KLADv7l3kK3ppMyCuLKoF0
fd7Ai2KW5ToIwzFofvJcS/STa6HA5gQenRUCAwEAAQJBAIq9amn00aS0h/CrjXqu
/ThglAXJmZhOMPVn4eiu7/ROixi9sex436MaVeMqSNf7Ex9a8fRNfWss7Sqd9eWu
RTUCIQDasvGASLqmjeffBNLTXV2A5g4t+kLVCpsEIZAycV5GswIhANEPLmax0ME/
EO+ZJ79TJKN5yiGBRsv5yvx5UiHxajEXAiAhAol5N4EUyq6I9w1rYdhPMGpLfk7A
IU2snfRJ6Nq2CQIgFrPsWRCkV+gOYcajD17rEqmuLrdIRexpg8N1DOSXoJ8CIGlS
tAboUGBxTDq3ZroNism3DaMIbKPyYrAqhKov1h5V
-----END RSA PRIVATE KEY-----
`

func bigFromString(s string) *big.Int {
	ret := new(big.Int);
	ret.SetString(s, 10);
	return ret;
}

var rsaPrivateKey = &rsa.PrivateKey{
	PublicKey: rsa.PublicKey{
		N: bigFromString("9353930466774385905609975137998169297361893554149986716853295022578535724979677252958524466350471210367835187480748268864277464700638583474144061408845077"),
		E: 65537,
	},
	D: bigFromString("7266398431328116344057699379749222532279343923819063639497049039389899328538543087657733766554155839834519529439851673014800261285757759040931985506583861"),
	P: bigFromString("98920366548084643601728869055592650835572950932266967461790948584315647051443"),
	Q: bigFromString("94560208308847015747498523884063394671606671904944666360068158221458669711639"),
}

*/

