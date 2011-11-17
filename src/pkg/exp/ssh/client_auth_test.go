// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"bytes"
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/pem"
	"errors"
	"io"
	"io/ioutil"
	"testing"
)

const _pem = `-----BEGIN RSA PRIVATE KEY-----
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

// reused internally by tests
var serverConfig = new(ServerConfig)

func init() {
	if err := serverConfig.SetRSAPrivateKey([]byte(_pem)); err != nil {
		panic("unable to set private key: " + err.Error())
	}
}

// keychain implements the ClientPublickey interface
type keychain struct {
	keys []*rsa.PrivateKey
}

func (k *keychain) Key(i int) (interface{}, error) {
	if i < 0 || i >= len(k.keys) {
		return nil, nil
	}
	return k.keys[i].PublicKey, nil
}

func (k *keychain) Sign(i int, rand io.Reader, data []byte) (sig []byte, err error) {
	hashFunc := crypto.SHA1
	h := hashFunc.New()
	h.Write(data)
	digest := h.Sum()
	return rsa.SignPKCS1v15(rand, k.keys[i], hashFunc, digest)
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

var pkey *rsa.PrivateKey

func init() {
	var err error
	pkey, err = rsa.GenerateKey(rand.Reader, 512)
	if err != nil {
		panic("unable to generate public key")
	}
}

func TestClientAuthPublickey(t *testing.T) {
	k := new(keychain)
	k.keys = append(k.keys, pkey)

	serverConfig.PubKeyCallback = func(user, algo string, pubkey []byte) bool {
		expected := []byte(serializePublickey(k.keys[0].PublicKey))
		algoname := algoName(k.keys[0].PublicKey)
		return user == "testuser" && algo == algoname && bytes.Equal(pubkey, expected)
	}
	serverConfig.PasswordCallback = nil

	l, err := Listen("tcp", "127.0.0.1:0", serverConfig)
	if err != nil {
		t.Fatalf("unable to listen: %s", err)
	}
	defer l.Close()

	done := make(chan bool, 1)
	go func() {
		c, err := l.Accept()
		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()
		if err := c.Handshake(); err != nil {
			t.Error(err)
		}
		done <- true
	}()

	config := &ClientConfig{
		User: "testuser",
		Auth: []ClientAuth{
			ClientAuthPublickey(k),
		},
	}

	c, err := Dial("tcp", l.Addr().String(), config)
	if err != nil {
		t.Fatalf("unable to dial remote side: %s", err)
	}
	defer c.Close()
	<-done
}

// password implements the ClientPassword interface
type password string

func (p password) Password(user string) (string, error) {
	return string(p), nil
}

func TestClientAuthPassword(t *testing.T) {
	pw := password("tiger")

	serverConfig.PasswordCallback = func(user, pass string) bool {
		return user == "testuser" && pass == string(pw)
	}
	serverConfig.PubKeyCallback = nil

	l, err := Listen("tcp", "127.0.0.1:0", serverConfig)
	if err != nil {
		t.Fatalf("unable to listen: %s", err)
	}
	defer l.Close()

	done := make(chan bool)
	go func() {
		c, err := l.Accept()
		if err != nil {
			t.Fatal(err)
		}
		if err := c.Handshake(); err != nil {
			t.Error(err)
		}
		defer c.Close()
		done <- true
	}()

	config := &ClientConfig{
		User: "testuser",
		Auth: []ClientAuth{
			ClientAuthPassword(pw),
		},
	}

	c, err := Dial("tcp", l.Addr().String(), config)
	if err != nil {
		t.Fatalf("unable to dial remote side: %s", err)
	}
	defer c.Close()
	<-done
}

func TestClientAuthPasswordAndPublickey(t *testing.T) {
	pw := password("tiger")

	serverConfig.PasswordCallback = func(user, pass string) bool {
		return user == "testuser" && pass == string(pw)
	}

	k := new(keychain)
	k.keys = append(k.keys, pkey)

	serverConfig.PubKeyCallback = func(user, algo string, pubkey []byte) bool {
		expected := []byte(serializePublickey(k.keys[0].PublicKey))
		algoname := algoName(k.keys[0].PublicKey)
		return user == "testuser" && algo == algoname && bytes.Equal(pubkey, expected)
	}

	l, err := Listen("tcp", "127.0.0.1:0", serverConfig)
	if err != nil {
		t.Fatalf("unable to listen: %s", err)
	}
	defer l.Close()

	done := make(chan bool)
	go func() {
		c, err := l.Accept()
		if err != nil {
			t.Fatal(err)
		}
		if err := c.Handshake(); err != nil {
			t.Error(err)
		}
		defer c.Close()
		done <- true
	}()

	wrongPw := password("wrong")
	config := &ClientConfig{
		User: "testuser",
		Auth: []ClientAuth{
			ClientAuthPassword(wrongPw),
			ClientAuthPublickey(k),
		},
	}

	c, err := Dial("tcp", l.Addr().String(), config)
	if err != nil {
		t.Fatalf("unable to dial remote side: %s", err)
	}
	defer c.Close()
	<-done
}
