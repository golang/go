// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"bytes"
	"crypto"
	"crypto/dsa"
	"crypto/rsa"
	_ "crypto/sha1"
	"crypto/x509"
	"encoding/pem"
	"errors"
	"io"
	"io/ioutil"
	"math/big"
	"testing"
)

// private key for mock server
const testServerPrivateKey = `-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA19lGVsTqIT5iiNYRgnoY1CwkbETW5cq+Rzk5v/kTlf31XpSU
70HVWkbTERECjaYdXM2gGcbb+sxpq6GtXf1M3kVomycqhxwhPv4Cr6Xp4WT/jkFx
9z+FFzpeodGJWjOH6L2H5uX1Cvr9EDdQp9t9/J32/qBFntY8GwoUI/y/1MSTmMiF
tupdMODN064vd3gyMKTwrlQ8tZM6aYuyOPsutLlUY7M5x5FwMDYvnPDSeyT/Iw0z
s3B+NCyqeeMd2T7YzQFnRATj0M7rM5LoSs7DVqVriOEABssFyLj31PboaoLhOKgc
qoM9khkNzr7FHVvi+DhYM2jD0DwvqZLN6NmnLwIDAQABAoIBAQCGVj+kuSFOV1lT
+IclQYA6bM6uY5mroqcSBNegVxCNhWU03BxlW//BE9tA/+kq53vWylMeN9mpGZea
riEMIh25KFGWXqXlOOioH8bkMsqA8S7sBmc7jljyv+0toQ9vCCtJ+sueNPhxQQxH
D2YvUjfzBQ04I9+wn30BByDJ1QA/FoPsunxIOUCcRBE/7jxuLYcpR+JvEF68yYIh
atXRld4W4in7T65YDR8jK1Uj9XAcNeDYNpT/M6oFLx1aPIlkG86aCWRO19S1jLPT
b1ZAKHHxPMCVkSYW0RqvIgLXQOR62D0Zne6/2wtzJkk5UCjkSQ2z7ZzJpMkWgDgN
ifCULFPBAoGBAPoMZ5q1w+zB+knXUD33n1J+niN6TZHJulpf2w5zsW+m2K6Zn62M
MXndXlVAHtk6p02q9kxHdgov34Uo8VpuNjbS1+abGFTI8NZgFo+bsDxJdItemwC4
KJ7L1iz39hRN/ZylMRLz5uTYRGddCkeIHhiG2h7zohH/MaYzUacXEEy3AoGBANz8
e/msleB+iXC0cXKwds26N4hyMdAFE5qAqJXvV3S2W8JZnmU+sS7vPAWMYPlERPk1
D8Q2eXqdPIkAWBhrx4RxD7rNc5qFNcQWEhCIxC9fccluH1y5g2M+4jpMX2CT8Uv+
3z+NoJ5uDTXZTnLCfoZzgZ4nCZVZ+6iU5U1+YXFJAoGBANLPpIV920n/nJmmquMj
orI1R/QXR9Cy56cMC65agezlGOfTYxk5Cfl5Ve+/2IJCfgzwJyjWUsFx7RviEeGw
64o7JoUom1HX+5xxdHPsyZ96OoTJ5RqtKKoApnhRMamau0fWydH1yeOEJd+TRHhc
XStGfhz8QNa1dVFvENczja1vAoGABGWhsd4VPVpHMc7lUvrf4kgKQtTC2PjA4xoc
QJ96hf/642sVE76jl+N6tkGMzGjnVm4P2j+bOy1VvwQavKGoXqJBRd5Apppv727g
/SM7hBXKFc/zH80xKBBgP/i1DR7kdjakCoeu4ngeGywvu2jTS6mQsqzkK+yWbUxJ
I7mYBsECgYB/KNXlTEpXtz/kwWCHFSYA8U74l7zZbVD8ul0e56JDK+lLcJ0tJffk
gqnBycHj6AhEycjda75cs+0zybZvN4x65KZHOGW/O/7OAWEcZP5TPb3zf9ned3Hl
NsZoFj52ponUM6+99A2CmezFCN16c4mbA//luWF+k3VVqR6BpkrhKw==
-----END RSA PRIVATE KEY-----`

const testClientPrivateKey = `-----BEGIN RSA PRIVATE KEY-----
MIIBOwIBAAJBALdGZxkXDAjsYk10ihwU6Id2KeILz1TAJuoq4tOgDWxEEGeTrcld
r/ZwVaFzjWzxaf6zQIJbfaSEAhqD5yo72+sCAwEAAQJBAK8PEVU23Wj8mV0QjwcJ
tZ4GcTUYQL7cF4+ezTCE9a1NrGnCP2RuQkHEKxuTVrxXt+6OF15/1/fuXnxKjmJC
nxkCIQDaXvPPBi0c7vAxGwNY9726x01/dNbHCE0CBtcotobxpwIhANbbQbh3JHVW
2haQh4fAG5mhesZKAGcxTyv4mQ7uMSQdAiAj+4dzMpJWdSzQ+qGHlHMIBvVHLkqB
y2VdEyF7DPCZewIhAI7GOI/6LDIFOvtPo6Bj2nNmyQ1HU6k/LRtNIXi4c9NJAiAr
rrxx26itVhJmcvoUhOjwuzSlP2bE5VHAvkGB352YBg==
-----END RSA PRIVATE KEY-----`

// keychain implements the ClientPublickey interface
type keychain struct {
	keys []interface{}
}

func (k *keychain) Key(i int) (interface{}, error) {
	if i < 0 || i >= len(k.keys) {
		return nil, nil
	}
	switch key := k.keys[i].(type) {
	case *rsa.PrivateKey:
		return key.PublicKey, nil
	case *dsa.PrivateKey:
		return key.PublicKey, nil
	}
	panic("unknown key type")
}

func (k *keychain) Sign(i int, rand io.Reader, data []byte) (sig []byte, err error) {
	hashFunc := crypto.SHA1
	h := hashFunc.New()
	h.Write(data)
	digest := h.Sum(nil)
	switch key := k.keys[i].(type) {
	case *rsa.PrivateKey:
		return rsa.SignPKCS1v15(rand, key, hashFunc, digest)
	}
	return nil, errors.New("unknown key type")
}

func (k *keychain) loadPEM(file string) error {
	buf, err := ioutil.ReadFile(file)
	if err != nil {
		return err
	}
	block, _ := pem.Decode(buf)
	if block == nil {
		return errors.New("ssh: no key found")
	}
	r, err := x509.ParsePKCS1PrivateKey(block.Bytes)
	if err != nil {
		return err
	}
	k.keys = append(k.keys, r)
	return nil
}

// password implements the ClientPassword interface
type password string

func (p password) Password(user string) (string, error) {
	return string(p), nil
}

// reused internally by tests
var (
	rsakey         *rsa.PrivateKey
	dsakey         *dsa.PrivateKey
	clientKeychain = new(keychain)
	clientPassword = password("tiger")
	serverConfig   = &ServerConfig{
		PasswordCallback: func(user, pass string) bool {
			return user == "testuser" && pass == string(clientPassword)
		},
		PubKeyCallback: func(user, algo string, pubkey []byte) bool {
			key := clientKeychain.keys[0].(*rsa.PrivateKey).PublicKey
			expected := []byte(serializePublickey(key))
			algoname := algoName(key)
			return user == "testuser" && algo == algoname && bytes.Equal(pubkey, expected)
		},
	}
)

func init() {
	if err := serverConfig.SetRSAPrivateKey([]byte(testServerPrivateKey)); err != nil {
		panic("unable to set private key: " + err.Error())
	}

	block, _ := pem.Decode([]byte(testClientPrivateKey))
	rsakey, _ = x509.ParsePKCS1PrivateKey(block.Bytes)

	clientKeychain.keys = append(clientKeychain.keys, rsakey)
	dsakey = new(dsa.PrivateKey)
	// taken from crypto/dsa/dsa_test.go
	dsakey.P, _ = new(big.Int).SetString("A9B5B793FB4785793D246BAE77E8FF63CA52F442DA763C440259919FE1BC1D6065A9350637A04F75A2F039401D49F08E066C4D275A5A65DA5684BC563C14289D7AB8A67163BFBF79D85972619AD2CFF55AB0EE77A9002B0EF96293BDD0F42685EBB2C66C327079F6C98000FBCB79AACDE1BC6F9D5C7B1A97E3D9D54ED7951FEF", 16)
	dsakey.Q, _ = new(big.Int).SetString("E1D3391245933D68A0714ED34BBCB7A1F422B9C1", 16)
	dsakey.G, _ = new(big.Int).SetString("634364FC25248933D01D1993ECABD0657CC0CB2CEED7ED2E3E8AECDFCDC4A25C3B15E9E3B163ACA2984B5539181F3EFF1A5E8903D71D5B95DA4F27202B77D2C44B430BB53741A8D59A8F86887525C9F2A6A5980A195EAA7F2FF910064301DEF89D3AA213E1FAC7768D89365318E370AF54A112EFBA9246D9158386BA1B4EEFDA", 16)
	dsakey.Y, _ = new(big.Int).SetString("32969E5780CFE1C849A1C276D7AEB4F38A23B591739AA2FE197349AEEBD31366AEE5EB7E6C6DDB7C57D02432B30DB5AA66D9884299FAA72568944E4EEDC92EA3FBC6F39F53412FBCC563208F7C15B737AC8910DBC2D9C9B8C001E72FDC40EB694AB1F06A5A2DBD18D9E36C66F31F566742F11EC0A52E9F7B89355C02FB5D32D2", 16)
	dsakey.X, _ = new(big.Int).SetString("5078D4D29795CBE76D3AACFE48C9AF0BCDBEE91A", 16)
}

// newMockAuthServer creates a new Server bound to 
// the loopback interface. The server exits after 
// processing one handshake.
func newMockAuthServer(t *testing.T) string {
	l, err := Listen("tcp", "127.0.0.1:0", serverConfig)
	if err != nil {
		t.Fatalf("unable to newMockAuthServer: %s", err)
	}
	go func() {
		defer l.Close()
		c, err := l.Accept()
		defer c.Close()
		if err != nil {
			t.Errorf("Unable to accept incoming connection: %v", err)
			return
		}
		if err := c.Handshake(); err != nil {
			// not Errorf because this is expected to
			// fail for some tests.
			t.Logf("Handshaking error: %v", err)
			return
		}
	}()
	return l.Addr().String()
}

func TestClientAuthPublickey(t *testing.T) {
	config := &ClientConfig{
		User: "testuser",
		Auth: []ClientAuth{
			ClientAuthPublickey(clientKeychain),
		},
	}
	c, err := Dial("tcp", newMockAuthServer(t), config)
	if err != nil {
		t.Fatalf("unable to dial remote side: %s", err)
	}
	c.Close()
}

func TestClientAuthPassword(t *testing.T) {
	config := &ClientConfig{
		User: "testuser",
		Auth: []ClientAuth{
			ClientAuthPassword(clientPassword),
		},
	}

	c, err := Dial("tcp", newMockAuthServer(t), config)
	if err != nil {
		t.Fatalf("unable to dial remote side: %s", err)
	}
	c.Close()
}

func TestClientAuthWrongPassword(t *testing.T) {
	wrongPw := password("wrong")
	config := &ClientConfig{
		User: "testuser",
		Auth: []ClientAuth{
			ClientAuthPassword(wrongPw),
			ClientAuthPublickey(clientKeychain),
		},
	}

	c, err := Dial("tcp", newMockAuthServer(t), config)
	if err != nil {
		t.Fatalf("unable to dial remote side: %s", err)
	}
	c.Close()
}

// the mock server will only authenticate ssh-rsa keys
func TestClientAuthInvalidPublickey(t *testing.T) {
	kc := new(keychain)
	kc.keys = append(kc.keys, dsakey)
	config := &ClientConfig{
		User: "testuser",
		Auth: []ClientAuth{
			ClientAuthPublickey(kc),
		},
	}

	c, err := Dial("tcp", newMockAuthServer(t), config)
	if err == nil {
		c.Close()
		t.Fatalf("dsa private key should not have authenticated with rsa public key")
	}
}

// the client should authenticate with the second key
func TestClientAuthRSAandDSA(t *testing.T) {
	kc := new(keychain)
	kc.keys = append(kc.keys, dsakey, rsakey)
	config := &ClientConfig{
		User: "testuser",
		Auth: []ClientAuth{
			ClientAuthPublickey(kc),
		},
	}
	c, err := Dial("tcp", newMockAuthServer(t), config)
	if err != nil {
		t.Fatalf("client could not authenticate with rsa key: %v", err)
	}
	c.Close()
}
