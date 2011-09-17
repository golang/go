// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"bufio"
	"crypto"
	"crypto/aes"
	"crypto/cipher"
	"crypto/hmac"
	"crypto/subtle"
	"hash"
	"io"
	"net"
	"os"
)

// halfConnection represents one direction of an SSH connection. It maintains
// the cipher state needed to process messages.
type halfConnection struct {
	// Only one of these two will be non-nil
	in  *bufio.Reader
	out net.Conn

	rand            io.Reader
	cipherAlgo      string
	macAlgo         string
	compressionAlgo string
	paddingMultiple int

	seqNum uint32

	mac    hash.Hash
	cipher cipher.Stream
}

func (hc *halfConnection) readOnePacket() (packet []byte, err os.Error) {
	var lengthBytes [5]byte

	_, err = io.ReadFull(hc.in, lengthBytes[:])
	if err != nil {
		return
	}

	if hc.cipher != nil {
		hc.cipher.XORKeyStream(lengthBytes[:], lengthBytes[:])
	}

	macSize := 0
	if hc.mac != nil {
		hc.mac.Reset()
		var seqNumBytes [4]byte
		seqNumBytes[0] = byte(hc.seqNum >> 24)
		seqNumBytes[1] = byte(hc.seqNum >> 16)
		seqNumBytes[2] = byte(hc.seqNum >> 8)
		seqNumBytes[3] = byte(hc.seqNum)
		hc.mac.Write(seqNumBytes[:])
		hc.mac.Write(lengthBytes[:])
		macSize = hc.mac.Size()
	}

	length := uint32(lengthBytes[0])<<24 | uint32(lengthBytes[1])<<16 | uint32(lengthBytes[2])<<8 | uint32(lengthBytes[3])

	paddingLength := uint32(lengthBytes[4])

	if length <= paddingLength+1 {
		return nil, os.NewError("invalid packet length")
	}
	if length > maxPacketSize {
		return nil, os.NewError("packet too large")
	}

	packet = make([]byte, length-1+uint32(macSize))
	_, err = io.ReadFull(hc.in, packet)
	if err != nil {
		return nil, err
	}
	mac := packet[length-1:]
	if hc.cipher != nil {
		hc.cipher.XORKeyStream(packet, packet[:length-1])
	}

	if hc.mac != nil {
		hc.mac.Write(packet[:length-1])
		if subtle.ConstantTimeCompare(hc.mac.Sum(), mac) != 1 {
			return nil, os.NewError("ssh: MAC failure")
		}
	}

	hc.seqNum++
	packet = packet[:length-paddingLength-1]
	return
}

func (hc *halfConnection) readPacket() (packet []byte, err os.Error) {
	for {
		packet, err := hc.readOnePacket()
		if err != nil {
			return nil, err
		}
		if packet[0] != msgIgnore && packet[0] != msgDebug {
			return packet, nil
		}
	}
	panic("unreachable")
}

func (hc *halfConnection) writePacket(packet []byte) os.Error {
	paddingMultiple := hc.paddingMultiple
	if paddingMultiple == 0 {
		paddingMultiple = 8
	}

	paddingLength := paddingMultiple - (4+1+len(packet))%paddingMultiple
	if paddingLength < 4 {
		paddingLength += paddingMultiple
	}

	var lengthBytes [5]byte
	length := len(packet) + 1 + paddingLength
	lengthBytes[0] = byte(length >> 24)
	lengthBytes[1] = byte(length >> 16)
	lengthBytes[2] = byte(length >> 8)
	lengthBytes[3] = byte(length)
	lengthBytes[4] = byte(paddingLength)

	var padding [32]byte
	_, err := io.ReadFull(hc.rand, padding[:paddingLength])
	if err != nil {
		return err
	}

	if hc.mac != nil {
		hc.mac.Reset()
		var seqNumBytes [4]byte
		seqNumBytes[0] = byte(hc.seqNum >> 24)
		seqNumBytes[1] = byte(hc.seqNum >> 16)
		seqNumBytes[2] = byte(hc.seqNum >> 8)
		seqNumBytes[3] = byte(hc.seqNum)
		hc.mac.Write(seqNumBytes[:])
		hc.mac.Write(lengthBytes[:])
		hc.mac.Write(packet)
		hc.mac.Write(padding[:paddingLength])
	}

	if hc.cipher != nil {
		hc.cipher.XORKeyStream(lengthBytes[:], lengthBytes[:])
		hc.cipher.XORKeyStream(packet, packet)
		hc.cipher.XORKeyStream(padding[:], padding[:paddingLength])
	}

	_, err = hc.out.Write(lengthBytes[:])
	if err != nil {
		return err
	}
	_, err = hc.out.Write(packet)
	if err != nil {
		return err
	}
	_, err = hc.out.Write(padding[:paddingLength])
	if err != nil {
		return err
	}

	if hc.mac != nil {
		_, err = hc.out.Write(hc.mac.Sum())
	}

	hc.seqNum++

	return err
}

const (
	serverKeys = iota
	clientKeys
)

// setupServerKeys sets the cipher and MAC keys from K, H and sessionId, as
// described in RFC 4253, section 6.4. direction should either be serverKeys
// (to setup server->client keys) or clientKeys (for client->server keys).
func (hc *halfConnection) setupKeys(direction int, K, H, sessionId []byte, hashFunc crypto.Hash) os.Error {
	h := hashFunc.New()

	// We only support these algorithms for now.
	if hc.cipherAlgo != cipherAES128CTR || hc.macAlgo != macSHA196 {
		return os.NewError("ssh: setupServerKeys internal error")
	}

	blockSize := 16
	keySize := 16
	macKeySize := 20

	var ivTag, keyTag, macKeyTag byte
	if direction == serverKeys {
		ivTag, keyTag, macKeyTag = 'B', 'D', 'F'
	} else {
		ivTag, keyTag, macKeyTag = 'A', 'C', 'E'
	}

	iv := make([]byte, blockSize)
	key := make([]byte, keySize)
	macKey := make([]byte, macKeySize)
	generateKeyMaterial(iv, ivTag, K, H, sessionId, h)
	generateKeyMaterial(key, keyTag, K, H, sessionId, h)
	generateKeyMaterial(macKey, macKeyTag, K, H, sessionId, h)

	hc.mac = truncatingMAC{12, hmac.NewSHA1(macKey)}
	aes, err := aes.NewCipher(key)
	if err != nil {
		return err
	}
	hc.cipher = cipher.NewCTR(aes, iv)
	hc.paddingMultiple = 16
	return nil
}

// generateKeyMaterial fills out with key material generated from tag, K, H
// and sessionId, as specified in RFC 4253, section 7.2.
func generateKeyMaterial(out []byte, tag byte, K, H, sessionId []byte, h hash.Hash) {
	var digestsSoFar []byte

	for len(out) > 0 {
		h.Reset()
		h.Write(K)
		h.Write(H)

		if len(digestsSoFar) == 0 {
			h.Write([]byte{tag})
			h.Write(sessionId)
		} else {
			h.Write(digestsSoFar)
		}

		digest := h.Sum()
		n := copy(out, digest)
		out = out[n:]
		if len(out) > 0 {
			digestsSoFar = append(digestsSoFar, digest...)
		}
	}
}

// truncatingMAC wraps around a hash.Hash and truncates the output digest to
// a given size.
type truncatingMAC struct {
	length int
	hmac   hash.Hash
}

func (t truncatingMAC) Write(data []byte) (int, os.Error) {
	return t.hmac.Write(data)
}

func (t truncatingMAC) Sum() []byte {
	digest := t.hmac.Sum()
	return digest[:t.length]
}

func (t truncatingMAC) Reset() {
	t.hmac.Reset()
}

func (t truncatingMAC) Size() int {
	return t.length
}

// maxVersionStringBytes is the maximum number of bytes that we'll accept as a
// version string. In the event that the client is talking a different protocol
// we need to set a limit otherwise we will keep using more and more memory
// while searching for the end of the version handshake.
const maxVersionStringBytes = 1024

func readVersion(r *bufio.Reader) (versionString []byte, ok bool) {
	versionString = make([]byte, 0, 64)
	seenCR := false

forEachByte:
	for len(versionString) < maxVersionStringBytes {
		b, err := r.ReadByte()
		if err != nil {
			return
		}

		if !seenCR {
			if b == '\r' {
				seenCR = true
			}
		} else {
			if b == '\n' {
				ok = true
				break forEachByte
			} else {
				seenCR = false
			}
		}
		versionString = append(versionString, b)
	}

	if ok {
		// We need to remove the CR from versionString
		versionString = versionString[:len(versionString)-1]
	}

	return
}
