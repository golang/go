// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"crypto/dsa"
	"crypto/rsa"
	"math/big"
	"strconv"
	"sync"
)

// These are string constants in the SSH protocol.
const (
	kexAlgoDH14SHA1 = "diffie-hellman-group14-sha1"
	hostAlgoRSA     = "ssh-rsa"
	macSHA196       = "hmac-sha1-96"
	compressionNone = "none"
	serviceUserAuth = "ssh-userauth"
	serviceSSH      = "ssh-connection"
)

var supportedKexAlgos = []string{kexAlgoDH14SHA1}
var supportedHostKeyAlgos = []string{hostAlgoRSA}
var supportedMACs = []string{macSHA196}
var supportedCompressions = []string{compressionNone}

// dhGroup is a multiplicative group suitable for implementing Diffie-Hellman key agreement.
type dhGroup struct {
	g, p *big.Int
}

// dhGroup14 is the group called diffie-hellman-group14-sha1 in RFC 4253 and
// Oakley Group 14 in RFC 3526.
var dhGroup14 *dhGroup

var dhGroup14Once sync.Once

func initDHGroup14() {
	p, _ := new(big.Int).SetString("FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3BE39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA051015728E5A8AACAA68FFFFFFFFFFFFFFFF", 16)

	dhGroup14 = &dhGroup{
		g: new(big.Int).SetInt64(2),
		p: p,
	}
}

// UnexpectedMessageError results when the SSH message that we received didn't
// match what we wanted.
type UnexpectedMessageError struct {
	expected, got uint8
}

func (u UnexpectedMessageError) Error() string {
	return "ssh: unexpected message type " + strconv.Itoa(int(u.got)) + " (expected " + strconv.Itoa(int(u.expected)) + ")"
}

// ParseError results from a malformed SSH message.
type ParseError struct {
	msgType uint8
}

func (p ParseError) Error() string {
	return "ssh: parse error in message type " + strconv.Itoa(int(p.msgType))
}

type handshakeMagics struct {
	clientVersion, serverVersion []byte
	clientKexInit, serverKexInit []byte
}

func findCommonAlgorithm(clientAlgos []string, serverAlgos []string) (commonAlgo string, ok bool) {
	for _, clientAlgo := range clientAlgos {
		for _, serverAlgo := range serverAlgos {
			if clientAlgo == serverAlgo {
				return clientAlgo, true
			}
		}
	}

	return
}

func findAgreedAlgorithms(transport *transport, clientKexInit, serverKexInit *kexInitMsg) (kexAlgo, hostKeyAlgo string, ok bool) {
	kexAlgo, ok = findCommonAlgorithm(clientKexInit.KexAlgos, serverKexInit.KexAlgos)
	if !ok {
		return
	}

	hostKeyAlgo, ok = findCommonAlgorithm(clientKexInit.ServerHostKeyAlgos, serverKexInit.ServerHostKeyAlgos)
	if !ok {
		return
	}

	transport.writer.cipherAlgo, ok = findCommonAlgorithm(clientKexInit.CiphersClientServer, serverKexInit.CiphersClientServer)
	if !ok {
		return
	}

	transport.reader.cipherAlgo, ok = findCommonAlgorithm(clientKexInit.CiphersServerClient, serverKexInit.CiphersServerClient)
	if !ok {
		return
	}

	transport.writer.macAlgo, ok = findCommonAlgorithm(clientKexInit.MACsClientServer, serverKexInit.MACsClientServer)
	if !ok {
		return
	}

	transport.reader.macAlgo, ok = findCommonAlgorithm(clientKexInit.MACsServerClient, serverKexInit.MACsServerClient)
	if !ok {
		return
	}

	transport.writer.compressionAlgo, ok = findCommonAlgorithm(clientKexInit.CompressionClientServer, serverKexInit.CompressionClientServer)
	if !ok {
		return
	}

	transport.reader.compressionAlgo, ok = findCommonAlgorithm(clientKexInit.CompressionServerClient, serverKexInit.CompressionServerClient)
	if !ok {
		return
	}

	ok = true
	return
}

// Cryptographic configuration common to both ServerConfig and ClientConfig.
type CryptoConfig struct {
	// The allowed cipher algorithms. If unspecified then DefaultCipherOrder is
	// used.
	Ciphers []string
}

func (c *CryptoConfig) ciphers() []string {
	if c.Ciphers == nil {
		return DefaultCipherOrder
	}
	return c.Ciphers
}

// serialize a signed slice according to RFC 4254 6.6.
func serializeSignature(algoname string, sig []byte) []byte {
	length := stringLength([]byte(algoname))
	length += stringLength(sig)

	ret := make([]byte, length)
	r := marshalString(ret, []byte(algoname))
	r = marshalString(r, sig)

	return ret
}

// serialize an rsa.PublicKey or dsa.PublicKey according to RFC 4253 6.6.
func serializePublickey(key interface{}) []byte {
	algoname := algoName(key)
	switch key := key.(type) {
	case rsa.PublicKey:
		e := new(big.Int).SetInt64(int64(key.E))
		length := stringLength([]byte(algoname))
		length += intLength(e)
		length += intLength(key.N)
		ret := make([]byte, length)
		r := marshalString(ret, []byte(algoname))
		r = marshalInt(r, e)
		marshalInt(r, key.N)
		return ret
	case dsa.PublicKey:
		length := stringLength([]byte(algoname))
		length += intLength(key.P)
		length += intLength(key.Q)
		length += intLength(key.G)
		length += intLength(key.Y)
		ret := make([]byte, length)
		r := marshalString(ret, []byte(algoname))
		r = marshalInt(r, key.P)
		r = marshalInt(r, key.Q)
		r = marshalInt(r, key.G)
		marshalInt(r, key.Y)
		return ret
	}
	panic("unexpected key type")
}

func algoName(key interface{}) string {
	switch key.(type) {
	case rsa.PublicKey:
		return "ssh-rsa"
	case dsa.PublicKey:
		return "ssh-dss"
	}
	panic("unexpected key type")
}

// buildDataSignedForAuth returns the data that is signed in order to prove
// posession of a private key. See RFC 4252, section 7.
func buildDataSignedForAuth(sessionId []byte, req userAuthRequestMsg, algo, pubKey []byte) []byte {
	user := []byte(req.User)
	service := []byte(req.Service)
	method := []byte(req.Method)

	length := stringLength(sessionId)
	length += 1
	length += stringLength(user)
	length += stringLength(service)
	length += stringLength(method)
	length += 1
	length += stringLength(algo)
	length += stringLength(pubKey)

	ret := make([]byte, length)
	r := marshalString(ret, sessionId)
	r[0] = msgUserAuthRequest
	r = r[1:]
	r = marshalString(r, user)
	r = marshalString(r, service)
	r = marshalString(r, method)
	r[0] = 1
	r = r[1:]
	r = marshalString(r, algo)
	r = marshalString(r, pubKey)
	return ret
}

// safeString sanitises s according to RFC 4251, section 9.2. 
// All control characters except tab, carriage return and newline are
// replaced by 0x20.
func safeString(s string) string {
	out := []byte(s)
	for i, c := range out {
		if c < 0x20 && c != 0xd && c != 0xa && c != 0x9 {
			out[i] = 0x20
		}
	}
	return string(out)
}
