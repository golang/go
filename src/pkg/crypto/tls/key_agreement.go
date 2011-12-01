// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"crypto"
	"crypto/elliptic"
	"crypto/md5"
	"crypto/rsa"
	"crypto/sha1"
	"crypto/x509"
	"errors"
	"io"
	"math/big"
)

// rsaKeyAgreement implements the standard TLS key agreement where the client
// encrypts the pre-master secret to the server's public key.
type rsaKeyAgreement struct{}

func (ka rsaKeyAgreement) generateServerKeyExchange(config *Config, clientHello *clientHelloMsg, hello *serverHelloMsg) (*serverKeyExchangeMsg, error) {
	return nil, nil
}

func (ka rsaKeyAgreement) processClientKeyExchange(config *Config, ckx *clientKeyExchangeMsg, version uint16) ([]byte, error) {
	preMasterSecret := make([]byte, 48)
	_, err := io.ReadFull(config.rand(), preMasterSecret[2:])
	if err != nil {
		return nil, err
	}

	if len(ckx.ciphertext) < 2 {
		return nil, errors.New("bad ClientKeyExchange")
	}

	ciphertext := ckx.ciphertext
	if version != versionSSL30 {
		ciphertextLen := int(ckx.ciphertext[0])<<8 | int(ckx.ciphertext[1])
		if ciphertextLen != len(ckx.ciphertext)-2 {
			return nil, errors.New("bad ClientKeyExchange")
		}
		ciphertext = ckx.ciphertext[2:]
	}

	err = rsa.DecryptPKCS1v15SessionKey(config.rand(), config.Certificates[0].PrivateKey, ciphertext, preMasterSecret)
	if err != nil {
		return nil, err
	}
	// We don't check the version number in the premaster secret.  For one,
	// by checking it, we would leak information about the validity of the
	// encrypted pre-master secret. Secondly, it provides only a small
	// benefit against a downgrade attack and some implementations send the
	// wrong version anyway. See the discussion at the end of section
	// 7.4.7.1 of RFC 4346.
	return preMasterSecret, nil
}

func (ka rsaKeyAgreement) processServerKeyExchange(config *Config, clientHello *clientHelloMsg, serverHello *serverHelloMsg, cert *x509.Certificate, skx *serverKeyExchangeMsg) error {
	return errors.New("unexpected ServerKeyExchange")
}

func (ka rsaKeyAgreement) generateClientKeyExchange(config *Config, clientHello *clientHelloMsg, cert *x509.Certificate) ([]byte, *clientKeyExchangeMsg, error) {
	preMasterSecret := make([]byte, 48)
	preMasterSecret[0] = byte(clientHello.vers >> 8)
	preMasterSecret[1] = byte(clientHello.vers)
	_, err := io.ReadFull(config.rand(), preMasterSecret[2:])
	if err != nil {
		return nil, nil, err
	}

	encrypted, err := rsa.EncryptPKCS1v15(config.rand(), cert.PublicKey.(*rsa.PublicKey), preMasterSecret)
	if err != nil {
		return nil, nil, err
	}
	ckx := new(clientKeyExchangeMsg)
	ckx.ciphertext = make([]byte, len(encrypted)+2)
	ckx.ciphertext[0] = byte(len(encrypted) >> 8)
	ckx.ciphertext[1] = byte(len(encrypted))
	copy(ckx.ciphertext[2:], encrypted)
	return preMasterSecret, ckx, nil
}

// md5SHA1Hash implements TLS 1.0's hybrid hash function which consists of the
// concatenation of an MD5 and SHA1 hash.
func md5SHA1Hash(slices ...[]byte) []byte {
	md5sha1 := make([]byte, md5.Size+sha1.Size)
	hmd5 := md5.New()
	for _, slice := range slices {
		hmd5.Write(slice)
	}
	copy(md5sha1, hmd5.Sum(nil))

	hsha1 := sha1.New()
	for _, slice := range slices {
		hsha1.Write(slice)
	}
	copy(md5sha1[md5.Size:], hsha1.Sum(nil))
	return md5sha1
}

// ecdheRSAKeyAgreement implements a TLS key agreement where the server
// generates a ephemeral EC public/private key pair and signs it. The
// pre-master secret is then calculated using ECDH.
type ecdheRSAKeyAgreement struct {
	privateKey []byte
	curve      *elliptic.Curve
	x, y       *big.Int
}

func (ka *ecdheRSAKeyAgreement) generateServerKeyExchange(config *Config, clientHello *clientHelloMsg, hello *serverHelloMsg) (*serverKeyExchangeMsg, error) {
	var curveid uint16

Curve:
	for _, c := range clientHello.supportedCurves {
		switch c {
		case curveP256:
			ka.curve = elliptic.P256()
			curveid = c
			break Curve
		case curveP384:
			ka.curve = elliptic.P384()
			curveid = c
			break Curve
		case curveP521:
			ka.curve = elliptic.P521()
			curveid = c
			break Curve
		}
	}

	var x, y *big.Int
	var err error
	ka.privateKey, x, y, err = ka.curve.GenerateKey(config.rand())
	if err != nil {
		return nil, err
	}
	ecdhePublic := ka.curve.Marshal(x, y)

	// http://tools.ietf.org/html/rfc4492#section-5.4
	serverECDHParams := make([]byte, 1+2+1+len(ecdhePublic))
	serverECDHParams[0] = 3 // named curve
	serverECDHParams[1] = byte(curveid >> 8)
	serverECDHParams[2] = byte(curveid)
	serverECDHParams[3] = byte(len(ecdhePublic))
	copy(serverECDHParams[4:], ecdhePublic)

	md5sha1 := md5SHA1Hash(clientHello.random, hello.random, serverECDHParams)
	sig, err := rsa.SignPKCS1v15(config.rand(), config.Certificates[0].PrivateKey, crypto.MD5SHA1, md5sha1)
	if err != nil {
		return nil, errors.New("failed to sign ECDHE parameters: " + err.Error())
	}

	skx := new(serverKeyExchangeMsg)
	skx.key = make([]byte, len(serverECDHParams)+2+len(sig))
	copy(skx.key, serverECDHParams)
	k := skx.key[len(serverECDHParams):]
	k[0] = byte(len(sig) >> 8)
	k[1] = byte(len(sig))
	copy(k[2:], sig)

	return skx, nil
}

func (ka *ecdheRSAKeyAgreement) processClientKeyExchange(config *Config, ckx *clientKeyExchangeMsg, version uint16) ([]byte, error) {
	if len(ckx.ciphertext) == 0 || int(ckx.ciphertext[0]) != len(ckx.ciphertext)-1 {
		return nil, errors.New("bad ClientKeyExchange")
	}
	x, y := ka.curve.Unmarshal(ckx.ciphertext[1:])
	if x == nil {
		return nil, errors.New("bad ClientKeyExchange")
	}
	x, _ = ka.curve.ScalarMult(x, y, ka.privateKey)
	preMasterSecret := make([]byte, (ka.curve.BitSize+7)>>3)
	xBytes := x.Bytes()
	copy(preMasterSecret[len(preMasterSecret)-len(xBytes):], xBytes)

	return preMasterSecret, nil
}

var errServerKeyExchange = errors.New("invalid ServerKeyExchange")

func (ka *ecdheRSAKeyAgreement) processServerKeyExchange(config *Config, clientHello *clientHelloMsg, serverHello *serverHelloMsg, cert *x509.Certificate, skx *serverKeyExchangeMsg) error {
	if len(skx.key) < 4 {
		return errServerKeyExchange
	}
	if skx.key[0] != 3 { // named curve
		return errors.New("server selected unsupported curve")
	}
	curveid := uint16(skx.key[1])<<8 | uint16(skx.key[2])

	switch curveid {
	case curveP256:
		ka.curve = elliptic.P256()
	case curveP384:
		ka.curve = elliptic.P384()
	case curveP521:
		ka.curve = elliptic.P521()
	default:
		return errors.New("server selected unsupported curve")
	}

	publicLen := int(skx.key[3])
	if publicLen+4 > len(skx.key) {
		return errServerKeyExchange
	}
	ka.x, ka.y = ka.curve.Unmarshal(skx.key[4 : 4+publicLen])
	if ka.x == nil {
		return errServerKeyExchange
	}
	serverECDHParams := skx.key[:4+publicLen]

	sig := skx.key[4+publicLen:]
	if len(sig) < 2 {
		return errServerKeyExchange
	}
	sigLen := int(sig[0])<<8 | int(sig[1])
	if sigLen+2 != len(sig) {
		return errServerKeyExchange
	}
	sig = sig[2:]

	md5sha1 := md5SHA1Hash(clientHello.random, serverHello.random, serverECDHParams)
	return rsa.VerifyPKCS1v15(cert.PublicKey.(*rsa.PublicKey), crypto.MD5SHA1, md5sha1, sig)
}

func (ka *ecdheRSAKeyAgreement) generateClientKeyExchange(config *Config, clientHello *clientHelloMsg, cert *x509.Certificate) ([]byte, *clientKeyExchangeMsg, error) {
	if ka.curve == nil {
		return nil, nil, errors.New("missing ServerKeyExchange message")
	}
	priv, mx, my, err := ka.curve.GenerateKey(config.rand())
	if err != nil {
		return nil, nil, err
	}
	x, _ := ka.curve.ScalarMult(ka.x, ka.y, priv)
	preMasterSecret := make([]byte, (ka.curve.BitSize+7)>>3)
	xBytes := x.Bytes()
	copy(preMasterSecret[len(preMasterSecret)-len(xBytes):], xBytes)

	serialized := ka.curve.Marshal(mx, my)

	ckx := new(clientKeyExchangeMsg)
	ckx.ciphertext = make([]byte, 1+len(serialized))
	ckx.ciphertext[0] = byte(len(serialized))
	copy(ckx.ciphertext[1:], serialized)

	return preMasterSecret, ckx, nil
}
