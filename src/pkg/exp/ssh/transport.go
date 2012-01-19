// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"bufio"
	"crypto"
	"crypto/cipher"
	"crypto/hmac"
	"crypto/sha1"
	"crypto/subtle"
	"errors"
	"hash"
	"io"
	"net"
	"sync"
)

const (
	packetSizeMultiple = 16 // TODO(huin) this should be determined by the cipher.
	minPacketSize      = 16
	maxPacketSize      = 36000
	minPaddingSize     = 4 // TODO(huin) should this be configurable?
)

// filteredConn reduces the set of methods exposed when embeddeding
// a net.Conn inside ssh.transport.
// TODO(dfc) suggestions for a better name will be warmly received.
type filteredConn interface {
	// Close closes the connection.
	Close() error

	// LocalAddr returns the local network address.
	LocalAddr() net.Addr

	// RemoteAddr returns the remote network address.
	RemoteAddr() net.Addr
}

// Types implementing packetWriter provide the ability to send packets to
// an SSH peer.
type packetWriter interface {
	// Encrypt and send a packet of data to the remote peer.
	writePacket(packet []byte) error
}

// transport represents the SSH connection to the remote peer.
type transport struct {
	reader
	writer

	filteredConn
}

// reader represents the incoming connection state.
type reader struct {
	io.Reader
	common
}

// writer represnts the outgoing connection state.
type writer struct {
	*sync.Mutex // protects writer.Writer from concurrent writes
	*bufio.Writer
	rand io.Reader
	common
}

// common represents the cipher state needed to process messages in a single
// direction.
type common struct {
	seqNum uint32
	mac    hash.Hash
	cipher cipher.Stream

	cipherAlgo      string
	macAlgo         string
	compressionAlgo string
}

// Read and decrypt a single packet from the remote peer.
func (r *reader) readOnePacket() ([]byte, error) {
	var lengthBytes = make([]byte, 5)
	var macSize uint32
	if _, err := io.ReadFull(r, lengthBytes); err != nil {
		return nil, err
	}

	r.cipher.XORKeyStream(lengthBytes, lengthBytes)

	if r.mac != nil {
		r.mac.Reset()
		seqNumBytes := []byte{
			byte(r.seqNum >> 24),
			byte(r.seqNum >> 16),
			byte(r.seqNum >> 8),
			byte(r.seqNum),
		}
		r.mac.Write(seqNumBytes)
		r.mac.Write(lengthBytes)
		macSize = uint32(r.mac.Size())
	}

	length := uint32(lengthBytes[0])<<24 | uint32(lengthBytes[1])<<16 | uint32(lengthBytes[2])<<8 | uint32(lengthBytes[3])
	paddingLength := uint32(lengthBytes[4])

	if length <= paddingLength+1 {
		return nil, errors.New("invalid packet length")
	}
	if length > maxPacketSize {
		return nil, errors.New("packet too large")
	}

	packet := make([]byte, length-1+macSize)
	if _, err := io.ReadFull(r, packet); err != nil {
		return nil, err
	}
	mac := packet[length-1:]
	r.cipher.XORKeyStream(packet, packet[:length-1])

	if r.mac != nil {
		r.mac.Write(packet[:length-1])
		if subtle.ConstantTimeCompare(r.mac.Sum(nil), mac) != 1 {
			return nil, errors.New("ssh: MAC failure")
		}
	}

	r.seqNum++
	return packet[:length-paddingLength-1], nil
}

// Read and decrypt next packet discarding debug and noop messages.
func (t *transport) readPacket() ([]byte, error) {
	for {
		packet, err := t.readOnePacket()
		if err != nil {
			return nil, err
		}
		if packet[0] != msgIgnore && packet[0] != msgDebug {
			return packet, nil
		}
	}
	panic("unreachable")
}

// Encrypt and send a packet of data to the remote peer.
func (w *writer) writePacket(packet []byte) error {
	w.Mutex.Lock()
	defer w.Mutex.Unlock()

	paddingLength := packetSizeMultiple - (5+len(packet))%packetSizeMultiple
	if paddingLength < 4 {
		paddingLength += packetSizeMultiple
	}

	length := len(packet) + 1 + paddingLength
	lengthBytes := []byte{
		byte(length >> 24),
		byte(length >> 16),
		byte(length >> 8),
		byte(length),
		byte(paddingLength),
	}
	padding := make([]byte, paddingLength)
	_, err := io.ReadFull(w.rand, padding)
	if err != nil {
		return err
	}

	if w.mac != nil {
		w.mac.Reset()
		seqNumBytes := []byte{
			byte(w.seqNum >> 24),
			byte(w.seqNum >> 16),
			byte(w.seqNum >> 8),
			byte(w.seqNum),
		}
		w.mac.Write(seqNumBytes)
		w.mac.Write(lengthBytes)
		w.mac.Write(packet)
		w.mac.Write(padding)
	}

	// TODO(dfc) lengthBytes, packet and padding should be
	// subslices of a single buffer
	w.cipher.XORKeyStream(lengthBytes, lengthBytes)
	w.cipher.XORKeyStream(packet, packet)
	w.cipher.XORKeyStream(padding, padding)

	if _, err := w.Write(lengthBytes); err != nil {
		return err
	}
	if _, err := w.Write(packet); err != nil {
		return err
	}
	if _, err := w.Write(padding); err != nil {
		return err
	}

	if w.mac != nil {
		if _, err := w.Write(w.mac.Sum(nil)); err != nil {
			return err
		}
	}

	if err := w.Flush(); err != nil {
		return err
	}
	w.seqNum++
	return err
}

// Send a message to the remote peer
func (t *transport) sendMessage(typ uint8, msg interface{}) error {
	packet := marshal(typ, msg)
	return t.writePacket(packet)
}

func newTransport(conn net.Conn, rand io.Reader) *transport {
	return &transport{
		reader: reader{
			Reader: bufio.NewReader(conn),
			common: common{
				cipher: noneCipher{},
			},
		},
		writer: writer{
			Writer: bufio.NewWriter(conn),
			rand:   rand,
			Mutex:  new(sync.Mutex),
			common: common{
				cipher: noneCipher{},
			},
		},
		filteredConn: conn,
	}
}

type direction struct {
	ivTag     []byte
	keyTag    []byte
	macKeyTag []byte
}

// TODO(dfc) can this be made a constant ?
var (
	serverKeys = direction{[]byte{'B'}, []byte{'D'}, []byte{'F'}}
	clientKeys = direction{[]byte{'A'}, []byte{'C'}, []byte{'E'}}
)

// setupKeys sets the cipher and MAC keys from kex.K, kex.H and sessionId, as
// described in RFC 4253, section 6.4. direction should either be serverKeys
// (to setup server->client keys) or clientKeys (for client->server keys).
func (c *common) setupKeys(d direction, K, H, sessionId []byte, hashFunc crypto.Hash) error {
	cipherMode := cipherModes[c.cipherAlgo]

	macKeySize := 20

	iv := make([]byte, cipherMode.ivSize)
	key := make([]byte, cipherMode.keySize)
	macKey := make([]byte, macKeySize)

	h := hashFunc.New()
	generateKeyMaterial(iv, d.ivTag, K, H, sessionId, h)
	generateKeyMaterial(key, d.keyTag, K, H, sessionId, h)
	generateKeyMaterial(macKey, d.macKeyTag, K, H, sessionId, h)

	c.mac = truncatingMAC{12, hmac.New(sha1.New, macKey)}

	cipher, err := cipherMode.createCipher(key, iv)
	if err != nil {
		return err
	}

	c.cipher = cipher

	return nil
}

// generateKeyMaterial fills out with key material generated from tag, K, H
// and sessionId, as specified in RFC 4253, section 7.2.
func generateKeyMaterial(out, tag []byte, K, H, sessionId []byte, h hash.Hash) {
	var digestsSoFar []byte

	for len(out) > 0 {
		h.Reset()
		h.Write(K)
		h.Write(H)

		if len(digestsSoFar) == 0 {
			h.Write(tag)
			h.Write(sessionId)
		} else {
			h.Write(digestsSoFar)
		}

		digest := h.Sum(nil)
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

func (t truncatingMAC) Write(data []byte) (int, error) {
	return t.hmac.Write(data)
}

func (t truncatingMAC) Sum(in []byte) []byte {
	out := t.hmac.Sum(in)
	return out[:len(in)+t.length]
}

func (t truncatingMAC) Reset() {
	t.hmac.Reset()
}

func (t truncatingMAC) Size() int {
	return t.length
}

func (t truncatingMAC) BlockSize() int { return t.hmac.BlockSize() }

// maxVersionStringBytes is the maximum number of bytes that we'll accept as a
// version string. In the event that the client is talking a different protocol
// we need to set a limit otherwise we will keep using more and more memory
// while searching for the end of the version handshake.
const maxVersionStringBytes = 1024

// Read version string as specified by RFC 4253, section 4.2.
func readVersion(r io.Reader) ([]byte, error) {
	versionString := make([]byte, 0, 64)
	var ok bool
	var buf [1]byte
forEachByte:
	for len(versionString) < maxVersionStringBytes {
		_, err := io.ReadFull(r, buf[:])
		if err != nil {
			return nil, err
		}
		// The RFC says that the version should be terminated with \r\n
		// but several SSH servers actually only send a \n.
		if buf[0] == '\n' {
			ok = true
			break forEachByte
		}
		versionString = append(versionString, buf[0])
	}

	if !ok {
		return nil, errors.New("ssh: failed to read version string")
	}

	// There might be a '\r' on the end which we should remove.
	if len(versionString) > 0 && versionString[len(versionString)-1] == '\r' {
		versionString = versionString[:len(versionString)-1]
	}
	return versionString, nil
}
