// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"crypto"
	"crypto/md5"
	"crypto/rsa"
	"crypto/sha1"
	"crypto/x509"
	"errors"
	"io"
)

var errClientKeyExchange = errors.New("tls: invalid ClientKeyExchange message")
var errServerKeyExchange = errors.New("tls: invalid ServerKeyExchange message")

// rsaKeyAgreement implements the standard TLS key agreement where the client
// encrypts the pre-master secret to the server's public key.
type rsaKeyAgreement struct{}

func (ka rsaKeyAgreement) generateServerKeyExchange(config *Config, cert *Certificate, clientHello *clientHelloMsg, hello *serverHelloMsg) (*serverKeyExchangeMsg, error) {
	return nil, nil
}

func (ka rsaKeyAgreement) processClientKeyExchange(config *Config, cert *Certificate, ckx *clientKeyExchangeMsg, version uint16) ([]byte, error) {
	if len(ckx.ciphertext) < 2 {
		return nil, errClientKeyExchange
	}

	ciphertext := ckx.ciphertext
	if version != VersionSSL30 {
		ciphertextLen := int(ckx.ciphertext[0])<<8 | int(ckx.ciphertext[1])
		if ciphertextLen != len(ckx.ciphertext)-2 {
			return nil, errClientKeyExchange
		}
		ciphertext = ckx.ciphertext[2:]
	}
	priv, ok := cert.PrivateKey.(crypto.Decrypter)
	if !ok {
		return nil, errors.New("tls: certificate private key does not implement crypto.Decrypter")
	}
	// Perform constant time RSA PKCS#1 v1.5 decryption
	preMasterSecret, err := priv.Decrypt(config.rand(), ciphertext, &rsa.PKCS1v15DecryptOptions{SessionKeyLen: 48})
	if err != nil {
		return nil, err
	}
	// We don't check the version number in the premaster secret. For one,
	// by checking it, we would leak information about the validity of the
	// encrypted pre-master secret. Secondly, it provides only a small
	// benefit against a downgrade attack and some implementations send the
	// wrong version anyway. See the discussion at the end of section
	// 7.4.7.1 of RFC 4346.
	return preMasterSecret, nil
}

func (ka rsaKeyAgreement) processServerKeyExchange(config *Config, clientHello *clientHelloMsg, serverHello *serverHelloMsg, cert *x509.Certificate, skx *serverKeyExchangeMsg) error {
	return errors.New("tls: unexpected ServerKeyExchange")
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

// sha1Hash calculates a SHA1 hash over the given byte slices.
func sha1Hash(slices [][]byte) []byte {
	hsha1 := sha1.New()
	for _, slice := range slices {
		hsha1.Write(slice)
	}
	return hsha1.Sum(nil)
}

// md5SHA1Hash implements TLS 1.0's hybrid hash function which consists of the
// concatenation of an MD5 and SHA1 hash.
func md5SHA1Hash(slices [][]byte) []byte {
	md5sha1 := make([]byte, md5.Size+sha1.Size)
	hmd5 := md5.New()
	for _, slice := range slices {
		hmd5.Write(slice)
	}
	copy(md5sha1, hmd5.Sum(nil))
	copy(md5sha1[md5.Size:], sha1Hash(slices))
	return md5sha1
}

// hashForServerKeyExchange hashes the given slices and returns their digest
// using the given hash function (for >= TLS 1.2) or using a default based on
// the sigType (for earlier TLS versions). For Ed25519 signatures, which don't
// do pre-hashing, it returns the concatenation of the slices.
func hashForServerKeyExchange(sigType uint8, hashFunc crypto.Hash, version uint16, slices ...[]byte) []byte {
	if sigType == signatureEd25519 {
		var signed []byte
		for _, slice := range slices {
			signed = append(signed, slice...)
		}
		return signed
	}
	if version >= VersionTLS12 {
		h := hashFunc.New()
		for _, slice := range slices {
			h.Write(slice)
		}
		digest := h.Sum(nil)
		return digest
	}
	if sigType == signatureECDSA {
		return sha1Hash(slices)
	}
	return md5SHA1Hash(slices)
}

// ecdheKeyAgreement implements a TLS key agreement where the server
// generates an ephemeral EC public/private key pair and signs it. The
// pre-master secret is then calculated using ECDH. The signature may
// be ECDSA, Ed25519 or RSA.
type ecdheKeyAgreement struct {
	version uint16
	isRSA   bool
	params  ecdheParameters

	// ckx and preMasterSecret are generated in processServerKeyExchange
	// and returned in generateClientKeyExchange.
	ckx             *clientKeyExchangeMsg
	preMasterSecret []byte
}

func (ka *ecdheKeyAgreement) generateServerKeyExchange(config *Config, cert *Certificate, clientHello *clientHelloMsg, hello *serverHelloMsg) (*serverKeyExchangeMsg, error) {
	preferredCurves := config.curvePreferences()

	var curveID CurveID
NextCandidate:
	for _, candidate := range preferredCurves {
		for _, c := range clientHello.supportedCurves {
			if candidate == c {
				curveID = c
				break NextCandidate
			}
		}
	}

	if curveID == 0 {
		return nil, errors.New("tls: no supported elliptic curves offered")
	}
	if _, ok := curveForCurveID(curveID); curveID != X25519 && !ok {
		return nil, errors.New("tls: CurvePreferences includes unsupported curve")
	}

	params, err := generateECDHEParameters(config.rand(), curveID)
	if err != nil {
		return nil, err
	}
	ka.params = params

	// See RFC 4492, Section 5.4.
	ecdhePublic := params.PublicKey()
	serverECDHParams := make([]byte, 1+2+1+len(ecdhePublic))
	serverECDHParams[0] = 3 // named curve
	serverECDHParams[1] = byte(curveID >> 8)
	serverECDHParams[2] = byte(curveID)
	serverECDHParams[3] = byte(len(ecdhePublic))
	copy(serverECDHParams[4:], ecdhePublic)

	priv, ok := cert.PrivateKey.(crypto.Signer)
	if !ok {
		return nil, errors.New("tls: certificate private key does not implement crypto.Signer")
	}

	signatureAlgorithm, sigType, hashFunc, err := pickSignatureAlgorithm(priv.Public(), clientHello.supportedSignatureAlgorithms, supportedSignatureAlgorithmsTLS12, ka.version)
	if err != nil {
		return nil, err
	}
	if (sigType == signaturePKCS1v15 || sigType == signatureRSAPSS) != ka.isRSA {
		return nil, errors.New("tls: certificate cannot be used with the selected cipher suite")
	}

	signed := hashForServerKeyExchange(sigType, hashFunc, ka.version, clientHello.random, hello.random, serverECDHParams)

	signOpts := crypto.SignerOpts(hashFunc)
	if sigType == signatureRSAPSS {
		signOpts = &rsa.PSSOptions{SaltLength: rsa.PSSSaltLengthEqualsHash, Hash: hashFunc}
	}
	sig, err := priv.Sign(config.rand(), signed, signOpts)
	if err != nil {
		return nil, errors.New("tls: failed to sign ECDHE parameters: " + err.Error())
	}

	skx := new(serverKeyExchangeMsg)
	sigAndHashLen := 0
	if ka.version >= VersionTLS12 {
		sigAndHashLen = 2
	}
	skx.key = make([]byte, len(serverECDHParams)+sigAndHashLen+2+len(sig))
	copy(skx.key, serverECDHParams)
	k := skx.key[len(serverECDHParams):]
	if ka.version >= VersionTLS12 {
		k[0] = byte(signatureAlgorithm >> 8)
		k[1] = byte(signatureAlgorithm)
		k = k[2:]
	}
	k[0] = byte(len(sig) >> 8)
	k[1] = byte(len(sig))
	copy(k[2:], sig)

	return skx, nil
}

func (ka *ecdheKeyAgreement) processClientKeyExchange(config *Config, cert *Certificate, ckx *clientKeyExchangeMsg, version uint16) ([]byte, error) {
	if len(ckx.ciphertext) == 0 || int(ckx.ciphertext[0]) != len(ckx.ciphertext)-1 {
		return nil, errClientKeyExchange
	}

	preMasterSecret := ka.params.SharedKey(ckx.ciphertext[1:])
	if preMasterSecret == nil {
		return nil, errClientKeyExchange
	}

	return preMasterSecret, nil
}

func (ka *ecdheKeyAgreement) processServerKeyExchange(config *Config, clientHello *clientHelloMsg, serverHello *serverHelloMsg, cert *x509.Certificate, skx *serverKeyExchangeMsg) error {
	if len(skx.key) < 4 {
		return errServerKeyExchange
	}
	if skx.key[0] != 3 { // named curve
		return errors.New("tls: server selected unsupported curve")
	}
	curveID := CurveID(skx.key[1])<<8 | CurveID(skx.key[2])

	publicLen := int(skx.key[3])
	if publicLen+4 > len(skx.key) {
		return errServerKeyExchange
	}
	serverECDHParams := skx.key[:4+publicLen]
	publicKey := serverECDHParams[4:]

	sig := skx.key[4+publicLen:]
	if len(sig) < 2 {
		return errServerKeyExchange
	}

	if _, ok := curveForCurveID(curveID); curveID != X25519 && !ok {
		return errors.New("tls: server selected unsupported curve")
	}

	params, err := generateECDHEParameters(config.rand(), curveID)
	if err != nil {
		return err
	}
	ka.params = params

	ka.preMasterSecret = params.SharedKey(publicKey)
	if ka.preMasterSecret == nil {
		return errServerKeyExchange
	}

	ourPublicKey := params.PublicKey()
	ka.ckx = new(clientKeyExchangeMsg)
	ka.ckx.ciphertext = make([]byte, 1+len(ourPublicKey))
	ka.ckx.ciphertext[0] = byte(len(ourPublicKey))
	copy(ka.ckx.ciphertext[1:], ourPublicKey)

	var signatureAlgorithm SignatureScheme
	if ka.version >= VersionTLS12 {
		// handle SignatureAndHashAlgorithm
		signatureAlgorithm = SignatureScheme(sig[0])<<8 | SignatureScheme(sig[1])
		sig = sig[2:]
		if len(sig) < 2 {
			return errServerKeyExchange
		}
	}
	_, sigType, hashFunc, err := pickSignatureAlgorithm(cert.PublicKey, []SignatureScheme{signatureAlgorithm}, clientHello.supportedSignatureAlgorithms, ka.version)
	if err != nil {
		return err
	}
	if (sigType == signaturePKCS1v15 || sigType == signatureRSAPSS) != ka.isRSA {
		return errServerKeyExchange
	}

	sigLen := int(sig[0])<<8 | int(sig[1])
	if sigLen+2 != len(sig) {
		return errServerKeyExchange
	}
	sig = sig[2:]

	signed := hashForServerKeyExchange(sigType, hashFunc, ka.version, clientHello.random, serverHello.random, serverECDHParams)
	return verifyHandshakeSignature(sigType, cert.PublicKey, hashFunc, signed, sig)
}

func (ka *ecdheKeyAgreement) generateClientKeyExchange(config *Config, clientHello *clientHelloMsg, cert *x509.Certificate) ([]byte, *clientKeyExchangeMsg, error) {
	if ka.ckx == nil {
		return nil, nil, errors.New("tls: missing ServerKeyExchange message")
	}

	return ka.preMasterSecret, ka.ckx, nil
}
