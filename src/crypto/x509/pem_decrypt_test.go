// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

import (
	"bytes"
	"crypto/rand"
	"encoding/base64"
	"encoding/pem"
	"strings"
	"testing"
)

func TestDecrypt(t *testing.T) {
	for i, data := range testData {
		t.Logf("test %v. %v", i, data.kind)
		block, rest := pem.Decode(data.pemData)
		if len(rest) > 0 {
			t.Error("extra data")
		}
		der, err := DecryptPEMBlock(block, data.password)
		if err != nil {
			t.Error("decrypt failed: ", err)
			continue
		}
		if _, err := ParsePKCS1PrivateKey(der); err != nil {
			t.Error("invalid private key: ", err)
		}
		plainDER, err := base64.StdEncoding.DecodeString(data.plainDER)
		if err != nil {
			t.Fatal("cannot decode test DER data: ", err)
		}
		if !bytes.Equal(der, plainDER) {
			t.Error("data mismatch")
		}
	}
}

func TestEncrypt(t *testing.T) {
	for i, data := range testData {
		t.Logf("test %v. %v", i, data.kind)
		plainDER, err := base64.StdEncoding.DecodeString(data.plainDER)
		if err != nil {
			t.Fatal("cannot decode test DER data: ", err)
		}
		password := []byte("kremvax1")
		block, err := EncryptPEMBlock(rand.Reader, "RSA PRIVATE KEY", plainDER, password, data.kind)
		if err != nil {
			t.Error("encrypt: ", err)
			continue
		}
		if !IsEncryptedPEMBlock(block) {
			t.Error("PEM block does not appear to be encrypted")
		}
		if block.Type != "RSA PRIVATE KEY" {
			t.Errorf("unexpected block type; got %q want %q", block.Type, "RSA PRIVATE KEY")
		}
		if block.Headers["Proc-Type"] != "4,ENCRYPTED" {
			t.Errorf("block does not have correct Proc-Type header")
		}
		der, err := DecryptPEMBlock(block, password)
		if err != nil {
			t.Error("decrypt: ", err)
			continue
		}
		if !bytes.Equal(der, plainDER) {
			t.Errorf("data mismatch")
		}
	}
}

var testData = []struct {
	kind     PEMCipher
	password []byte
	pemData  []byte
	plainDER string
}{
	{
		kind:     PEMCipherDES,
		password: []byte("asdf"),
		pemData: []byte(testingKey(`
-----BEGIN RSA TESTING KEY-----
Proc-Type: 4,ENCRYPTED
DEK-Info: DES-CBC,34F09A4FC8DE22B5

WXxy8kbZdiZvANtKvhmPBLV7eVFj2A5z6oAxvI9KGyhG0ZK0skfnt00C24vfU7m5
ICXeoqP67lzJ18xCzQfHjDaBNs53DSDT+Iz4e8QUep1xQ30+8QKX2NA2coee3nwc
6oM1cuvhNUDemBH2i3dKgMVkfaga0zQiiOq6HJyGSncCMSruQ7F9iWEfRbFcxFCx
qtHb1kirfGKEtgWTF+ynyco6+2gMXNu70L7nJcnxnV/RLFkHt7AUU1yrclxz7eZz
XOH9VfTjb52q/I8Suozq9coVQwg4tXfIoYUdT//O+mB7zJb9HI9Ps77b9TxDE6Gm
4C9brwZ3zg2vqXcwwV6QRZMtyll9rOpxkbw6NPlpfBqkc3xS51bbxivbO/Nve4KD
r12ymjFNF4stXCfJnNqKoZ50BHmEEUDu5Wb0fpVn82XrGw7CYc4iug==
-----END RSA TESTING KEY-----`)),
		plainDER: `
MIIBPAIBAAJBAPASZe+tCPU6p80AjHhDkVsLYa51D35e/YGa8QcZyooeZM8EHozo
KD0fNiKI+53bHdy07N+81VQ8/ejPcRoXPlsCAwEAAQJBAMTxIuSq27VpR+zZ7WJf
c6fvv1OBvpMZ0/d1pxL/KnOAgq2rD5hDtk9b0LGhTPgQAmrrMTKuSeGoIuYE+gKQ
QvkCIQD+GC1m+/do+QRurr0uo46Kx1LzLeSCrjBk34wiOp2+dwIhAPHfTLRXS2fv
7rljm0bYa4+eDZpz+E8RcXEgzhhvcQQ9AiAI5eHZJGOyml3MXnQjiPi55WcDOw0w
glcRgT6QCEtz2wIhANSyqaFtosIkHKqrDUGfz/bb5tqMYTAnBruVPaf/WEOBAiEA
9xORWeRG1tRpso4+dYy4KdDkuLPIO01KY6neYGm3BCM=`,
	},
	{
		kind:     PEMCipher3DES,
		password: []byte("asdf"),
		pemData: []byte(testingKey(`
-----BEGIN RSA TESTING KEY-----
Proc-Type: 4,ENCRYPTED
DEK-Info: DES-EDE3-CBC,C1F4A6A03682C2C7

0JqVdBEH6iqM7drTkj+e2W/bE3LqakaiWhb9WUVonFkhyu8ca/QzebY3b5gCvAZQ
YwBvDcT/GHospKqPx+cxDHJNsUASDZws6bz8ZXWJGwZGExKzr0+Qx5fgXn44Ms3x
8g1ENFuTXtxo+KoNK0zuAMAqp66Llcds3Fjl4XR18QaD0CrVNAfOdgATWZm5GJxk
Fgx5f84nT+/ovvreG+xeOzWgvtKo0UUZVrhGOgfKLpa57adumcJ6SkUuBtEFpZFB
ldw5w7WC7d13x2LsRkwo8ZrDKgIV+Y9GNvhuCCkTzNP0V3gNeJpd201HZHR+9n3w
3z0VjR/MGqsfcy1ziEWMNOO53At3zlG6zP05aHMnMcZoVXadEK6L1gz++inSSDCq
gI0UJP4e3JVB7AkgYymYAwiYALAkoEIuanxoc50njJk=
-----END RSA TESTING KEY-----`)),
		plainDER: `
MIIBOwIBAAJBANOCXKdoNS/iP/MAbl9cf1/SF3P+Ns7ZeNL27CfmDh0O6Zduaax5
NBiumd2PmjkaCu7lQ5JOibHfWn+xJsc3kw0CAwEAAQJANX/W8d1Q/sCqzkuAn4xl
B5a7qfJWaLHndu1QRLNTRJPn0Ee7OKJ4H0QKOhQM6vpjRrz+P2u9thn6wUxoPsef
QQIhAP/jCkfejFcy4v15beqKzwz08/tslVjF+Yq41eJGejmxAiEA05pMoqfkyjcx
fyvGhpoOyoCp71vSGUfR2I9CR65oKh0CIC1Msjs66LlfJtQctRq6bCEtFCxEcsP+
eEjYo/Sk6WphAiEAxpgWPMJeU/shFT28gS+tmhjPZLpEoT1qkVlC14u0b3ECIQDX
tZZZxCtPAm7shftEib0VU77Lk8MsXJcx2C4voRsjEw==`,
	},
	{
		kind:     PEMCipherAES128,
		password: []byte("asdf"),
		pemData: []byte(testingKey(`
-----BEGIN RSA TESTING KEY-----
Proc-Type: 4,ENCRYPTED
DEK-Info: AES-128-CBC,D4492E793FC835CC038A728ED174F78A

EyfQSzXSjv6BaNH+NHdXRlkHdimpF9izWlugVJAPApgXrq5YldPe2aGIOFXyJ+QE
ZIG20DYqaPzJRjTEbPNZ6Es0S2JJ5yCpKxwJuDkgJZKtF39Q2i36JeGbSZQIuWJE
GZbBpf1jDH/pr0iGonuAdl2PCCZUiy+8eLsD2tyviHUkFLOB+ykYoJ5t8ngZ/B6D
33U43LLb7+9zD4y3Q9OVHqBFGyHcxCY9+9Qh4ZnFp7DTf6RY5TNEvE3s4g6aDpBs
3NbvRVvYTgs8K9EPk4K+5R+P2kD8J8KvEIGxVa1vz8QoCJ/jr7Ka2rvNgPCex5/E
080LzLHPCrXKdlr/f50yhNWq08ZxMWQFkui+FDHPDUaEELKAXV8/5PDxw80Rtybo
AVYoCVIbZXZCuCO81op8UcOgEpTtyU5Lgh3Mw5scQL0=
-----END RSA TESTING KEY-----`)),
		plainDER: `
MIIBOgIBAAJBAMBlj5FxYtqbcy8wY89d/S7n0+r5MzD9F63BA/Lpl78vQKtdJ5dT
cDGh/rBt1ufRrNp0WihcmZi7Mpl/3jHjiWECAwEAAQJABNOHYnKhtDIqFYj1OAJ3
k3GlU0OlERmIOoeY/cL2V4lgwllPBEs7r134AY4wMmZSBUj8UR/O4SNO668ElKPE
cQIhAOuqY7/115x5KCdGDMWi+jNaMxIvI4ETGwV40ykGzqlzAiEA0P9oEC3m9tHB
kbpjSTxaNkrXxDgdEOZz8X0uOUUwHNsCIAwzcSCiGLyYJTULUmP1ESERfW1mlV78
XzzESaJpIM/zAiBQkSTcl9VhcJreQqvjn5BnPZLP4ZHS4gPwJAGdsj5J4QIhAOVR
B3WlRNTXR2WsJ5JdByezg9xzdXzULqmga0OE339a`,
	},
	{
		kind:     PEMCipherAES192,
		password: []byte("asdf"),
		pemData: []byte(testingKey(`
-----BEGIN RSA TESTING KEY-----
Proc-Type: 4,ENCRYPTED
DEK-Info: AES-192-CBC,E2C9FB02BCA23ADE1829F8D8BC5F5369

cqVslvHqDDM6qwU6YjezCRifXmKsrgEev7ng6Qs7UmDJOpHDgJQZI9fwMFUhIyn5
FbCu1SHkLMW52Ld3CuEqMnzWMlhPrW8tFvUOrMWPYSisv7nNq88HobZEJcUNL2MM
Y15XmHW6IJwPqhKyLHpWXyOCVEh4ODND2nV15PCoi18oTa475baxSk7+1qH7GuIs
Rb7tshNTMqHbCpyo9Rn3UxeFIf9efdl8YLiMoIqc7J8E5e9VlbeQSdLMQOgDAQJG
ReUtTw8exmKsY4gsSjhkg5uiw7/ZB1Ihto0qnfQJgjGc680qGkT1d6JfvOfeYAk6
xn5RqS/h8rYAYm64KnepfC9vIujo4NqpaREDmaLdX5MJPQ+SlytITQvgUsUq3q/t
Ss85xjQEZH3hzwjQqdJvmA4hYP6SUjxYpBM+02xZ1Xw=
-----END RSA TESTING KEY-----`)),
		plainDER: `
MIIBOwIBAAJBAMGcRrZiNNmtF20zyS6MQ7pdGx17aFDl+lTl+qnLuJRUCMUG05xs
OmxmL/O1Qlf+bnqR8Bgg65SfKg21SYuLhiMCAwEAAQJBAL94uuHyO4wux2VC+qpj
IzPykjdU7XRcDHbbvksf4xokSeUFjjD3PB0Qa83M94y89ZfdILIqS9x5EgSB4/lX
qNkCIQD6cCIqLfzq/lYbZbQgAAjpBXeQVYsbvVtJrPrXJAlVVQIhAMXpDKMeFPMn
J0g2rbx1gngx0qOa5r5iMU5w/noN4W2XAiBjf+WzCG5yFvazD+dOx3TC0A8+4x3P
uZ3pWbaXf5PNuQIgAcdXarvhelH2w2piY1g3BPeFqhzBSCK/yLGxR82KIh8CIQDD
+qGKsd09NhQ/G27y/DARzOYtml1NvdmCQAgsDIIOLA==`,
	},
	{
		kind:     PEMCipherAES256,
		password: []byte("asdf"),
		pemData: []byte(testingKey(`
-----BEGIN RSA TESTING KEY-----
Proc-Type: 4,ENCRYPTED
DEK-Info: AES-256-CBC,8E7ED5CD731902CE938957A886A5FFBD

4Mxr+KIzRVwoOP0wwq6caSkvW0iS+GE2h2Ov/u+n9ZTMwL83PRnmjfjzBgfRZLVf
JFPXxUK26kMNpIdssNnqGOds+DhB+oSrsNKoxgxSl5OBoYv9eJTVYm7qOyAFIsjr
DRKAcjYCmzfesr7PVTowwy0RtHmYwyXMGDlAzzZrEvaiySFFmMyKKvtoavwaFoc7
Pz3RZScwIuubzTGJ1x8EzdffYOsdCa9Mtgpp3L136+23dOd6L/qK2EG2fzrJSHs/
2XugkleBFSMKzEp9mxXKRfa++uidQvMZTFLDK9w5YjrRvMBo/l2BoZIsq0jAIE1N
sv5Z/KwlX+3MDEpPQpUwGPlGGdLnjI3UZ+cjgqBcoMiNc6HfgbBgYJSU6aDSHuCk
clCwByxWkBNgJ2GrkwNrF26v+bGJJJNR4SKouY1jQf0=
-----END RSA TESTING KEY-----`)),
		plainDER: `
MIIBOgIBAAJBAKy3GFkstoCHIEeUU/qO8207m8WSrjksR+p9B4tf1w5k+2O1V/GY
AQ5WFCApItcOkQe/I0yZZJk/PmCqMzSxrc8CAwEAAQJAOCAz0F7AW9oNelVQSP8F
Sfzx7O1yom+qWyAQQJF/gFR11gpf9xpVnnyu1WxIRnDUh1LZwUsjwlDYb7MB74id
oQIhANPcOiLwOPT4sIUpRM5HG6BF1BI7L77VpyGVk8xNP7X/AiEA0LMHZtk4I+lJ
nClgYp4Yh2JZ1Znbu7IoQMCEJCjwKDECIGd8Dzm5tViTkUW6Hs3Tlf73nNs65duF
aRnSglss8I3pAiEAonEnKruawgD8RavDFR+fUgmQiPz4FnGGeVgfwpGG1JECIBYq
PXHYtPqxQIbD2pScR5qum7iGUh11lEUPkmt+2uqS`,
	},
	{
		// generated with:
		// openssl genrsa -aes128 -passout pass:asdf -out server.orig.key 128
		kind:     PEMCipherAES128,
		password: []byte("asdf"),
		pemData: []byte(testingKey(`
-----BEGIN RSA TESTING KEY-----
Proc-Type: 4,ENCRYPTED
DEK-Info: AES-128-CBC,74611ABC2571AF11B1BF9B69E62C89E7

6ei/MlytjE0FFgZOGQ+jrwomKfpl8kdefeE0NSt/DMRrw8OacHAzBNi3pPEa0eX3
eND9l7C9meCirWovjj9QWVHrXyugFuDIqgdhQ8iHTgCfF3lrmcttVrbIfMDw+smD
hTP8O1mS/MHl92NE0nhv0w==
-----END RSA TESTING KEY-----`)),
		plainDER: `
MGMCAQACEQC6ssxmYuauuHGOCDAI54RdAgMBAAECEQCWIn6Yv2O+kBcDF7STctKB
AgkA8SEfu/2i3g0CCQDGNlXbBHX7kQIIK3Ww5o0cYbECCQDCimPb0dYGsQIIeQ7A
jryIst8=`,
	},
}

var incompleteBlockPEM = testingKey(`
-----BEGIN RSA TESTING KEY-----
Proc-Type: 4,ENCRYPTED
DEK-Info: AES-128-CBC,74611ABC2571AF11B1BF9B69E62C89E7

6L8yXK2MTQUWBk4ZD6OvCiYp+mXyR1594TQ1K38MxGvDw5pwcDME2Lek8RrR5fd40P2XsL2Z4KKt
ai+OP1BZUetfK6AW4MiqB2FDyIdOAJ8XeWuZy21Wtsh8wPD6yYOFM/w7WZL8weX3Y0TSeG/T
-----END RSA TESTING KEY-----`)

func TestIncompleteBlock(t *testing.T) {
	// incompleteBlockPEM contains ciphertext that is not a multiple of the
	// block size. This previously panicked. See #11215.
	block, _ := pem.Decode([]byte(incompleteBlockPEM))
	_, err := DecryptPEMBlock(block, []byte("foo"))
	if err == nil {
		t.Fatal("Bad PEM data decrypted successfully")
	}
	const expectedSubstr = "block size"
	if e := err.Error(); !strings.Contains(e, expectedSubstr) {
		t.Fatalf("Expected error containing %q but got: %q", expectedSubstr, e)
	}
}

func testingKey(s string) string { return strings.ReplaceAll(s, "TESTING KEY", "PRIVATE KEY") }
