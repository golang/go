// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"strconv"
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

// UnexpectedMessageError results when the SSH message that we received didn't
// match what we wanted.
type UnexpectedMessageError struct {
	expected, got uint8
}

func (u UnexpectedMessageError) String() string {
	return "ssh: unexpected message type " + strconv.Itoa(int(u.got)) + " (expected " + strconv.Itoa(int(u.expected)) + ")"
}

// ParseError results from a malformed SSH message.
type ParseError struct {
	msgType uint8
}

func (p ParseError) String() string {
	return "ssh: parse error in message type " + strconv.Itoa(int(p.msgType))
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

func findAgreedAlgorithms(clientToServer, serverToClient *transport, clientKexInit, serverKexInit *kexInitMsg) (kexAlgo, hostKeyAlgo string, ok bool) {
	kexAlgo, ok = findCommonAlgorithm(clientKexInit.KexAlgos, serverKexInit.KexAlgos)
	if !ok {
		return
	}

	hostKeyAlgo, ok = findCommonAlgorithm(clientKexInit.ServerHostKeyAlgos, serverKexInit.ServerHostKeyAlgos)
	if !ok {
		return
	}

	clientToServer.cipherAlgo, ok = findCommonAlgorithm(clientKexInit.CiphersClientServer, serverKexInit.CiphersClientServer)
	if !ok {
		return
	}

	serverToClient.cipherAlgo, ok = findCommonAlgorithm(clientKexInit.CiphersServerClient, serverKexInit.CiphersServerClient)
	if !ok {
		return
	}

	clientToServer.macAlgo, ok = findCommonAlgorithm(clientKexInit.MACsClientServer, serverKexInit.MACsClientServer)
	if !ok {
		return
	}

	serverToClient.macAlgo, ok = findCommonAlgorithm(clientKexInit.MACsServerClient, serverKexInit.MACsServerClient)
	if !ok {
		return
	}

	clientToServer.compressionAlgo, ok = findCommonAlgorithm(clientKexInit.CompressionClientServer, serverKexInit.CompressionClientServer)
	if !ok {
		return
	}

	serverToClient.compressionAlgo, ok = findCommonAlgorithm(clientKexInit.CompressionServerClient, serverKexInit.CompressionServerClient)
	if !ok {
		return
	}

	ok = true
	return
}
