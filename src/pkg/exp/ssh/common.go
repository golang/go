// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"math/big"
	"strconv"
	"sync"
)

// These are string constants in the SSH protocol.
const (
	kexAlgoDH14SHA1 = "diffie-hellman-group14-sha1"
	hostAlgoRSA     = "ssh-rsa"
	cipherAES128CTR = "aes128-ctr"
	macSHA196       = "hmac-sha1-96"
	compressionNone = "none"
	serviceUserAuth = "ssh-userauth"
	serviceSSH      = "ssh-connection"
)

var supportedKexAlgos = []string{kexAlgoDH14SHA1}
var supportedHostKeyAlgos = []string{hostAlgoRSA}
var supportedCiphers = []string{cipherAES128CTR}
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
