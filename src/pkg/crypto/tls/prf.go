// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"crypto/hmac";
	"crypto/md5";
	"crypto/sha1";
	"hash";
	"os";
	"strings";
)

// Split a premaster secret in two as specified in RFC 4346, section 5.
func splitPreMasterSecret(secret []byte) (s1, s2 []byte) {
	s1 = secret[0 : (len(secret)+1)/2];
	s2 = secret[len(secret)/2 : len(secret)];
	return;
}

// pHash implements the P_hash function, as defined in RFC 4346, section 5.
func pHash(result, secret, seed []byte, hash hash.Hash) {
	h := hmac.New(hash, secret);
	h.Write(seed);
	a := h.Sum();

	j := 0;
	for j < len(result) {
		h.Reset();
		h.Write(a);
		h.Write(seed);
		b := h.Sum();
		todo := len(b);
		if j+todo > len(result) {
			todo = len(result) - j
		}
		copy(result[j:j+todo], b);
		j += todo;

		h.Reset();
		h.Write(a);
		a = h.Sum();
	}
}

// pRF11 implements the TLS 1.1 pseudo-random function, as defined in RFC 4346, section 5.
func pRF11(result, secret, label, seed []byte) {
	hashSHA1 := sha1.New();
	hashMD5 := md5.New();

	labelAndSeed := make([]byte, len(label)+len(seed));
	copy(labelAndSeed, label);
	copy(labelAndSeed[len(label):len(labelAndSeed)], seed);

	s1, s2 := splitPreMasterSecret(secret);
	pHash(result, s1, labelAndSeed, hashMD5);
	result2 := make([]byte, len(result));
	pHash(result2, s2, labelAndSeed, hashSHA1);

	for i, b := range result2 {
		result[i] ^= b
	}
}

const (
	tlsRandomLength		= 32;	// Length of a random nonce in TLS 1.1.
	masterSecretLength	= 48;	// Length of a master secret in TLS 1.1.
	finishedVerifyLength	= 12;	// Length of verify_data in a Finished message.
)

var masterSecretLabel = strings.Bytes("master secret")
var keyExpansionLabel = strings.Bytes("key expansion")
var clientFinishedLabel = strings.Bytes("client finished")
var serverFinishedLabel = strings.Bytes("server finished")

// keysFromPreMasterSecret generates the connection keys from the pre master
// secret, given the lengths of the MAC and cipher keys, as defined in RFC
// 4346, section 6.3.
func keysFromPreMasterSecret11(preMasterSecret, clientRandom, serverRandom []byte, macLen, keyLen int) (masterSecret, clientMAC, serverMAC, clientKey, serverKey []byte) {
	var seed [tlsRandomLength * 2]byte;
	copy(seed[0:len(clientRandom)], clientRandom);
	copy(seed[len(clientRandom):len(seed)], serverRandom);
	masterSecret = make([]byte, masterSecretLength);
	pRF11(masterSecret, preMasterSecret, masterSecretLabel, seed[0:len(seed)]);

	copy(seed[0:len(clientRandom)], serverRandom);
	copy(seed[len(serverRandom):len(seed)], clientRandom);

	n := 2*macLen + 2*keyLen;
	keyMaterial := make([]byte, n);
	pRF11(keyMaterial, masterSecret, keyExpansionLabel, seed[0:len(seed)]);
	clientMAC = keyMaterial[0:macLen];
	serverMAC = keyMaterial[macLen : macLen*2];
	clientKey = keyMaterial[macLen*2 : macLen*2+keyLen];
	serverKey = keyMaterial[macLen*2+keyLen : len(keyMaterial)];
	return;
}

// A finishedHash calculates the hash of a set of handshake messages suitable
// for including in a Finished message.
type finishedHash struct {
	clientMD5	hash.Hash;
	clientSHA1	hash.Hash;
	serverMD5	hash.Hash;
	serverSHA1	hash.Hash;
}

func newFinishedHash() finishedHash {
	return finishedHash{md5.New(), sha1.New(), md5.New(), sha1.New()}
}

func (h finishedHash) Write(msg []byte) (n int, err os.Error) {
	h.clientMD5.Write(msg);
	h.clientSHA1.Write(msg);
	h.serverMD5.Write(msg);
	h.serverSHA1.Write(msg);
	return len(msg), nil;
}

// finishedSum calculates the contents of the verify_data member of a Finished
// message given the MD5 and SHA1 hashes of a set of handshake messages.
func finishedSum(md5, sha1, label, masterSecret []byte) []byte {
	seed := make([]byte, len(md5)+len(sha1));
	copy(seed, md5);
	copy(seed[len(md5):len(seed)], sha1);
	out := make([]byte, finishedVerifyLength);
	pRF11(out, masterSecret, label, seed);
	return out;
}

// clientSum returns the contents of the verify_data member of a client's
// Finished message.
func (h finishedHash) clientSum(masterSecret []byte) []byte {
	md5 := h.clientMD5.Sum();
	sha1 := h.clientSHA1.Sum();
	return finishedSum(md5, sha1, clientFinishedLabel, masterSecret);
}

// serverSum returns the contents of the verify_data member of a server's
// Finished message.
func (h finishedHash) serverSum(masterSecret []byte) []byte {
	md5 := h.serverMD5.Sum();
	sha1 := h.serverSHA1.Sum();
	return finishedSum(md5, sha1, serverFinishedLabel, masterSecret);
}
