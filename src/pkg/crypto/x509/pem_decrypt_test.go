// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

import (
	"encoding/pem"
	"testing"
)

func TestDecrypt(t *testing.T) {
	for _, data := range testData {
		block, rest := pem.Decode(data.pemData)
		if len(rest) > 0 {
			t.Error(data.kind, "extra data")
		}
		der, err := DecryptPEMBlock(block, data.password)
		if err != nil {
			t.Error(data.kind, err)
			continue
		}
		if _, err := ParsePKCS1PrivateKey(der); err != nil {
			t.Error(data.kind, "Invalid private key")
		}
	}
}

var testData = []struct {
	kind     string
	password []byte
	pemData  []byte
}{
	{
		kind:     "DES-CBC",
		password: []byte("asdf"),
		pemData: []byte(`
-----BEGIN RSA PRIVATE KEY-----
Proc-Type: 4,ENCRYPTED
DEK-Info: DES-CBC,34F09A4FC8DE22B5

WXxy8kbZdiZvANtKvhmPBLV7eVFj2A5z6oAxvI9KGyhG0ZK0skfnt00C24vfU7m5
ICXeoqP67lzJ18xCzQfHjDaBNs53DSDT+Iz4e8QUep1xQ30+8QKX2NA2coee3nwc
6oM1cuvhNUDemBH2i3dKgMVkfaga0zQiiOq6HJyGSncCMSruQ7F9iWEfRbFcxFCx
qtHb1kirfGKEtgWTF+ynyco6+2gMXNu70L7nJcnxnV/RLFkHt7AUU1yrclxz7eZz
XOH9VfTjb52q/I8Suozq9coVQwg4tXfIoYUdT//O+mB7zJb9HI9Ps77b9TxDE6Gm
4C9brwZ3zg2vqXcwwV6QRZMtyll9rOpxkbw6NPlpfBqkc3xS51bbxivbO/Nve4KD
r12ymjFNF4stXCfJnNqKoZ50BHmEEUDu5Wb0fpVn82XrGw7CYc4iug==
-----END RSA PRIVATE KEY-----`),
	},
	{
		kind:     "DES-EDE3-CBC",
		password: []byte("asdf"),
		pemData: []byte(`
-----BEGIN RSA PRIVATE KEY-----
Proc-Type: 4,ENCRYPTED
DEK-Info: DES-EDE3-CBC,C1F4A6A03682C2C7

0JqVdBEH6iqM7drTkj+e2W/bE3LqakaiWhb9WUVonFkhyu8ca/QzebY3b5gCvAZQ
YwBvDcT/GHospKqPx+cxDHJNsUASDZws6bz8ZXWJGwZGExKzr0+Qx5fgXn44Ms3x
8g1ENFuTXtxo+KoNK0zuAMAqp66Llcds3Fjl4XR18QaD0CrVNAfOdgATWZm5GJxk
Fgx5f84nT+/ovvreG+xeOzWgvtKo0UUZVrhGOgfKLpa57adumcJ6SkUuBtEFpZFB
ldw5w7WC7d13x2LsRkwo8ZrDKgIV+Y9GNvhuCCkTzNP0V3gNeJpd201HZHR+9n3w
3z0VjR/MGqsfcy1ziEWMNOO53At3zlG6zP05aHMnMcZoVXadEK6L1gz++inSSDCq
gI0UJP4e3JVB7AkgYymYAwiYALAkoEIuanxoc50njJk=
-----END RSA PRIVATE KEY-----`),
	},
	{
		kind:     "AES-128-CBC",
		password: []byte("asdf"),
		pemData: []byte(`
-----BEGIN RSA PRIVATE KEY-----
Proc-Type: 4,ENCRYPTED
DEK-Info: AES-128-CBC,D4492E793FC835CC038A728ED174F78A

EyfQSzXSjv6BaNH+NHdXRlkHdimpF9izWlugVJAPApgXrq5YldPe2aGIOFXyJ+QE
ZIG20DYqaPzJRjTEbPNZ6Es0S2JJ5yCpKxwJuDkgJZKtF39Q2i36JeGbSZQIuWJE
GZbBpf1jDH/pr0iGonuAdl2PCCZUiy+8eLsD2tyviHUkFLOB+ykYoJ5t8ngZ/B6D
33U43LLb7+9zD4y3Q9OVHqBFGyHcxCY9+9Qh4ZnFp7DTf6RY5TNEvE3s4g6aDpBs
3NbvRVvYTgs8K9EPk4K+5R+P2kD8J8KvEIGxVa1vz8QoCJ/jr7Ka2rvNgPCex5/E
080LzLHPCrXKdlr/f50yhNWq08ZxMWQFkui+FDHPDUaEELKAXV8/5PDxw80Rtybo
AVYoCVIbZXZCuCO81op8UcOgEpTtyU5Lgh3Mw5scQL0=
-----END RSA PRIVATE KEY-----`),
	},
	{
		kind:     "AES-192-CBC",
		password: []byte("asdf"),
		pemData: []byte(`
-----BEGIN RSA PRIVATE KEY-----
Proc-Type: 4,ENCRYPTED
DEK-Info: AES-192-CBC,E2C9FB02BCA23ADE1829F8D8BC5F5369

cqVslvHqDDM6qwU6YjezCRifXmKsrgEev7ng6Qs7UmDJOpHDgJQZI9fwMFUhIyn5
FbCu1SHkLMW52Ld3CuEqMnzWMlhPrW8tFvUOrMWPYSisv7nNq88HobZEJcUNL2MM
Y15XmHW6IJwPqhKyLHpWXyOCVEh4ODND2nV15PCoi18oTa475baxSk7+1qH7GuIs
Rb7tshNTMqHbCpyo9Rn3UxeFIf9efdl8YLiMoIqc7J8E5e9VlbeQSdLMQOgDAQJG
ReUtTw8exmKsY4gsSjhkg5uiw7/ZB1Ihto0qnfQJgjGc680qGkT1d6JfvOfeYAk6
xn5RqS/h8rYAYm64KnepfC9vIujo4NqpaREDmaLdX5MJPQ+SlytITQvgUsUq3q/t
Ss85xjQEZH3hzwjQqdJvmA4hYP6SUjxYpBM+02xZ1Xw=
-----END RSA PRIVATE KEY-----`),
	},
	{
		kind:     "AES-256-CBC",
		password: []byte("asdf"),
		pemData: []byte(`
-----BEGIN RSA PRIVATE KEY-----
Proc-Type: 4,ENCRYPTED
DEK-Info: AES-256-CBC,8E7ED5CD731902CE938957A886A5FFBD

4Mxr+KIzRVwoOP0wwq6caSkvW0iS+GE2h2Ov/u+n9ZTMwL83PRnmjfjzBgfRZLVf
JFPXxUK26kMNpIdssNnqGOds+DhB+oSrsNKoxgxSl5OBoYv9eJTVYm7qOyAFIsjr
DRKAcjYCmzfesr7PVTowwy0RtHmYwyXMGDlAzzZrEvaiySFFmMyKKvtoavwaFoc7
Pz3RZScwIuubzTGJ1x8EzdffYOsdCa9Mtgpp3L136+23dOd6L/qK2EG2fzrJSHs/
2XugkleBFSMKzEp9mxXKRfa++uidQvMZTFLDK9w5YjrRvMBo/l2BoZIsq0jAIE1N
sv5Z/KwlX+3MDEpPQpUwGPlGGdLnjI3UZ+cjgqBcoMiNc6HfgbBgYJSU6aDSHuCk
clCwByxWkBNgJ2GrkwNrF26v+bGJJJNR4SKouY1jQf0=
-----END RSA PRIVATE KEY-----`),
	},
	{
		// generated with:
		// openssl genrsa -aes128 -passout pass:asdf -out server.orig.key 128
		kind:     "AES-128-CBC",
		password: []byte("asdf"),
		pemData: []byte(`
-----BEGIN RSA PRIVATE KEY-----
Proc-Type: 4,ENCRYPTED
DEK-Info: AES-128-CBC,74611ABC2571AF11B1BF9B69E62C89E7

6ei/MlytjE0FFgZOGQ+jrwomKfpl8kdefeE0NSt/DMRrw8OacHAzBNi3pPEa0eX3
eND9l7C9meCirWovjj9QWVHrXyugFuDIqgdhQ8iHTgCfF3lrmcttVrbIfMDw+smD
hTP8O1mS/MHl92NE0nhv0w==
-----END RSA PRIVATE KEY-----`),
	},
}
