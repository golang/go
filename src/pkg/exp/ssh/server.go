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
	"math/big"
	"net"
	"sync"
)

type ServerConfig struct {
	rsa           *rsa.PrivateKey
	rsaSerialized []byte

	// Rand provides the source of entropy for key exchange. If Rand is 
	// nil, the cryptographic random reader in package crypto/rand will 
	// be used.
	Rand io.Reader

	// NoClientAuth is true if clients are allowed to connect without
	// authenticating.
	NoClientAuth bool

	// PasswordCallback, if non-nil, is called when a user attempts to
	// authenticate using a password. It may be called concurrently from
	// several goroutines.
	PasswordCallback func(user, password string) bool

	// PubKeyCallback, if non-nil, is called when a client attempts public
	// key authentication. It must return true iff the given public key is
	// valid for the given user.
	PubKeyCallback func(user, algo string, pubkey []byte) bool
}

func (c *ServerConfig) rand() io.Reader {
	if c.Rand == nil {
		return rand.Reader
	}
	return c.Rand
}

// SetRSAPrivateKey sets the private key for a Server. A Server must have a
// private key configured in order to accept connections. The private key must
// be in the form of a PEM encoded, PKCS#1, RSA private key. The file "id_rsa"
// typically contains such a key.
func (s *ServerConfig) SetRSAPrivateKey(pemBytes []byte) error {
	block, _ := pem.Decode(pemBytes)
	if block == nil {
		return errors.New("ssh: no key found")
	}
	var err error
	s.rsa, err = x509.ParsePKCS1PrivateKey(block.Bytes)
	if err != nil {
		return err
	}

	s.rsaSerialized = marshalRSA(s.rsa)
	return nil
}

// marshalRSA serializes an RSA private key according to RFC 4256, section 6.6.
func marshalRSA(priv *rsa.PrivateKey) []byte {
	e := new(big.Int).SetInt64(int64(priv.E))
	length := stringLength([]byte(hostAlgoRSA))
	length += intLength(e)
	length += intLength(priv.N)

	ret := make([]byte, length)
	r := marshalString(ret, []byte(hostAlgoRSA))
	r = marshalInt(r, e)
	r = marshalInt(r, priv.N)

	return ret
}

// parseRSA parses an RSA key according to RFC 4256, section 6.6.
func parseRSA(in []byte) (pubKey *rsa.PublicKey, ok bool) {
	algo, in, ok := parseString(in)
	if !ok || string(algo) != hostAlgoRSA {
		return nil, false
	}
	bigE, in, ok := parseInt(in)
	if !ok || bigE.BitLen() > 24 {
		return nil, false
	}
	e := bigE.Int64()
	if e < 3 || e&1 == 0 {
		return nil, false
	}
	N, in, ok := parseInt(in)
	if !ok || len(in) > 0 {
		return nil, false
	}
	return &rsa.PublicKey{
		N: N,
		E: int(e),
	}, true
}

func parseRSASig(in []byte) (sig []byte, ok bool) {
	algo, in, ok := parseString(in)
	if !ok || string(algo) != hostAlgoRSA {
		return nil, false
	}
	sig, in, ok = parseString(in)
	if len(in) > 0 {
		ok = false
	}
	return
}

// cachedPubKey contains the results of querying whether a public key is
// acceptable for a user. The cache only applies to a single ServerConn.
type cachedPubKey struct {
	user, algo string
	pubKey     []byte
	result     bool
}

const maxCachedPubKeys = 16

// A ServerConn represents an incomming connection.
type ServerConn struct {
	*transport
	config *ServerConfig

	channels   map[uint32]*channel
	nextChanId uint32

	// lock protects err and also allows Channels to serialise their writes
	// to out.
	lock sync.RWMutex
	err  error

	// cachedPubKeys contains the cache results of tests for public keys.
	// Since SSH clients will query whether a public key is acceptable
	// before attempting to authenticate with it, we end up with duplicate
	// queries for public key validity.
	cachedPubKeys []cachedPubKey
}

// Server returns a new SSH server connection
// using c as the underlying transport.
func Server(c net.Conn, config *ServerConfig) *ServerConn {
	conn := &ServerConn{
		transport: newTransport(c, config.rand()),
		channels:  make(map[uint32]*channel),
		config:    config,
	}
	return conn
}

// kexDH performs Diffie-Hellman key agreement on a ServerConnection. The
// returned values are given the same names as in RFC 4253, section 8.
func (s *ServerConn) kexDH(group *dhGroup, hashFunc crypto.Hash, magics *handshakeMagics, hostKeyAlgo string) (H, K []byte, err error) {
	packet, err := s.readPacket()
	if err != nil {
		return
	}
	var kexDHInit kexDHInitMsg
	if err = unmarshal(&kexDHInit, packet, msgKexDHInit); err != nil {
		return
	}

	if kexDHInit.X.Sign() == 0 || kexDHInit.X.Cmp(group.p) >= 0 {
		return nil, nil, errors.New("client DH parameter out of bounds")
	}

	y, err := rand.Int(s.config.rand(), group.p)
	if err != nil {
		return
	}

	Y := new(big.Int).Exp(group.g, y, group.p)
	kInt := new(big.Int).Exp(kexDHInit.X, y, group.p)

	var serializedHostKey []byte
	switch hostKeyAlgo {
	case hostAlgoRSA:
		serializedHostKey = s.config.rsaSerialized
	default:
		return nil, nil, errors.New("internal error")
	}

	h := hashFunc.New()
	writeString(h, magics.clientVersion)
	writeString(h, magics.serverVersion)
	writeString(h, magics.clientKexInit)
	writeString(h, magics.serverKexInit)
	writeString(h, serializedHostKey)
	writeInt(h, kexDHInit.X)
	writeInt(h, Y)
	K = make([]byte, intLength(kInt))
	marshalInt(K, kInt)
	h.Write(K)

	H = h.Sum()

	h.Reset()
	h.Write(H)
	hh := h.Sum()

	var sig []byte
	switch hostKeyAlgo {
	case hostAlgoRSA:
		sig, err = rsa.SignPKCS1v15(s.config.rand(), s.config.rsa, hashFunc, hh)
		if err != nil {
			return
		}
	default:
		return nil, nil, errors.New("internal error")
	}

	serializedSig := serializeRSASignature(sig)

	kexDHReply := kexDHReplyMsg{
		HostKey:   serializedHostKey,
		Y:         Y,
		Signature: serializedSig,
	}
	packet = marshal(msgKexDHReply, kexDHReply)

	err = s.writePacket(packet)
	return
}

func serializeRSASignature(sig []byte) []byte {
	length := stringLength([]byte(hostAlgoRSA))
	length += stringLength(sig)

	ret := make([]byte, length)
	r := marshalString(ret, []byte(hostAlgoRSA))
	r = marshalString(r, sig)

	return ret
}

// serverVersion is the fixed identification string that Server will use.
var serverVersion = []byte("SSH-2.0-Go\r\n")

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

// Handshake performs an SSH transport and client authentication on the given ServerConn.
func (s *ServerConn) Handshake() error {
	var magics handshakeMagics
	if _, err := s.Write(serverVersion); err != nil {
		return err
	}
	if err := s.Flush(); err != nil {
		return err
	}
	magics.serverVersion = serverVersion[:len(serverVersion)-2]

	version, err := readVersion(s)
	if err != nil {
		return err
	}
	magics.clientVersion = version

	serverKexInit := kexInitMsg{
		KexAlgos:                supportedKexAlgos,
		ServerHostKeyAlgos:      supportedHostKeyAlgos,
		CiphersClientServer:     supportedCiphers,
		CiphersServerClient:     supportedCiphers,
		MACsClientServer:        supportedMACs,
		MACsServerClient:        supportedMACs,
		CompressionClientServer: supportedCompressions,
		CompressionServerClient: supportedCompressions,
	}
	kexInitPacket := marshal(msgKexInit, serverKexInit)
	magics.serverKexInit = kexInitPacket

	if err := s.writePacket(kexInitPacket); err != nil {
		return err
	}

	packet, err := s.readPacket()
	if err != nil {
		return err
	}

	magics.clientKexInit = packet

	var clientKexInit kexInitMsg
	if err = unmarshal(&clientKexInit, packet, msgKexInit); err != nil {
		return err
	}

	kexAlgo, hostKeyAlgo, ok := findAgreedAlgorithms(s.transport, &clientKexInit, &serverKexInit)
	if !ok {
		return errors.New("ssh: no common algorithms")
	}

	if clientKexInit.FirstKexFollows && kexAlgo != clientKexInit.KexAlgos[0] {
		// The client sent a Kex message for the wrong algorithm,
		// which we have to ignore.
		if _, err := s.readPacket(); err != nil {
			return err
		}
	}

	var H, K []byte
	var hashFunc crypto.Hash
	switch kexAlgo {
	case kexAlgoDH14SHA1:
		hashFunc = crypto.SHA1
		dhGroup14Once.Do(initDHGroup14)
		H, K, err = s.kexDH(dhGroup14, hashFunc, &magics, hostKeyAlgo)
	default:
		err = errors.New("ssh: unexpected key exchange algorithm " + kexAlgo)
	}
	if err != nil {
		return err
	}

	if err = s.writePacket([]byte{msgNewKeys}); err != nil {
		return err
	}
	if err = s.transport.writer.setupKeys(serverKeys, K, H, H, hashFunc); err != nil {
		return err
	}
	if packet, err = s.readPacket(); err != nil {
		return err
	}

	if packet[0] != msgNewKeys {
		return UnexpectedMessageError{msgNewKeys, packet[0]}
	}
	s.transport.reader.setupKeys(clientKeys, K, H, H, hashFunc)
	if packet, err = s.readPacket(); err != nil {
		return err
	}

	var serviceRequest serviceRequestMsg
	if err = unmarshal(&serviceRequest, packet, msgServiceRequest); err != nil {
		return err
	}
	if serviceRequest.Service != serviceUserAuth {
		return errors.New("ssh: requested service '" + serviceRequest.Service + "' before authenticating")
	}
	serviceAccept := serviceAcceptMsg{
		Service: serviceUserAuth,
	}
	if err = s.writePacket(marshal(msgServiceAccept, serviceAccept)); err != nil {
		return err
	}

	if err = s.authenticate(H); err != nil {
		return err
	}
	return nil
}

func isAcceptableAlgo(algo string) bool {
	return algo == hostAlgoRSA
}

// testPubKey returns true if the given public key is acceptable for the user.
func (s *ServerConn) testPubKey(user, algo string, pubKey []byte) bool {
	if s.config.PubKeyCallback == nil || !isAcceptableAlgo(algo) {
		return false
	}

	for _, c := range s.cachedPubKeys {
		if c.user == user && c.algo == algo && bytes.Equal(c.pubKey, pubKey) {
			return c.result
		}
	}

	result := s.config.PubKeyCallback(user, algo, pubKey)
	if len(s.cachedPubKeys) < maxCachedPubKeys {
		c := cachedPubKey{
			user:   user,
			algo:   algo,
			pubKey: make([]byte, len(pubKey)),
			result: result,
		}
		copy(c.pubKey, pubKey)
		s.cachedPubKeys = append(s.cachedPubKeys, c)
	}

	return result
}

func (s *ServerConn) authenticate(H []byte) error {
	var userAuthReq userAuthRequestMsg
	var err error
	var packet []byte

userAuthLoop:
	for {
		if packet, err = s.readPacket(); err != nil {
			return err
		}
		if err = unmarshal(&userAuthReq, packet, msgUserAuthRequest); err != nil {
			return err
		}

		if userAuthReq.Service != serviceSSH {
			return errors.New("ssh: client attempted to negotiate for unknown service: " + userAuthReq.Service)
		}

		switch userAuthReq.Method {
		case "none":
			if s.config.NoClientAuth {
				break userAuthLoop
			}
		case "password":
			if s.config.PasswordCallback == nil {
				break
			}
			payload := userAuthReq.Payload
			if len(payload) < 1 || payload[0] != 0 {
				return ParseError{msgUserAuthRequest}
			}
			payload = payload[1:]
			password, payload, ok := parseString(payload)
			if !ok || len(payload) > 0 {
				return ParseError{msgUserAuthRequest}
			}

			if s.config.PasswordCallback(userAuthReq.User, string(password)) {
				break userAuthLoop
			}
		case "publickey":
			if s.config.PubKeyCallback == nil {
				break
			}
			payload := userAuthReq.Payload
			if len(payload) < 1 {
				return ParseError{msgUserAuthRequest}
			}
			isQuery := payload[0] == 0
			payload = payload[1:]
			algoBytes, payload, ok := parseString(payload)
			if !ok {
				return ParseError{msgUserAuthRequest}
			}
			algo := string(algoBytes)

			pubKey, payload, ok := parseString(payload)
			if !ok {
				return ParseError{msgUserAuthRequest}
			}
			if isQuery {
				// The client can query if the given public key
				// would be ok.
				if len(payload) > 0 {
					return ParseError{msgUserAuthRequest}
				}
				if s.testPubKey(userAuthReq.User, algo, pubKey) {
					okMsg := userAuthPubKeyOkMsg{
						Algo:   algo,
						PubKey: string(pubKey),
					}
					if err = s.writePacket(marshal(msgUserAuthPubKeyOk, okMsg)); err != nil {
						return err
					}
					continue userAuthLoop
				}
			} else {
				sig, payload, ok := parseString(payload)
				if !ok || len(payload) > 0 {
					return ParseError{msgUserAuthRequest}
				}
				if !isAcceptableAlgo(algo) {
					break
				}
				rsaSig, ok := parseRSASig(sig)
				if !ok {
					return ParseError{msgUserAuthRequest}
				}
				signedData := buildDataSignedForAuth(H, userAuthReq, algoBytes, pubKey)
				switch algo {
				case hostAlgoRSA:
					hashFunc := crypto.SHA1
					h := hashFunc.New()
					h.Write(signedData)
					digest := h.Sum()
					rsaKey, ok := parseRSA(pubKey)
					if !ok {
						return ParseError{msgUserAuthRequest}
					}
					if rsa.VerifyPKCS1v15(rsaKey, hashFunc, digest, rsaSig) != nil {
						return ParseError{msgUserAuthRequest}
					}
				default:
					return errors.New("ssh: isAcceptableAlgo incorrect")
				}
				if s.testPubKey(userAuthReq.User, algo, pubKey) {
					break userAuthLoop
				}
			}
		}

		var failureMsg userAuthFailureMsg
		if s.config.PasswordCallback != nil {
			failureMsg.Methods = append(failureMsg.Methods, "password")
		}
		if s.config.PubKeyCallback != nil {
			failureMsg.Methods = append(failureMsg.Methods, "publickey")
		}

		if len(failureMsg.Methods) == 0 {
			return errors.New("ssh: no authentication methods configured but NoClientAuth is also false")
		}

		if err = s.writePacket(marshal(msgUserAuthFailure, failureMsg)); err != nil {
			return err
		}
	}

	packet = []byte{msgUserAuthSuccess}
	if err = s.writePacket(packet); err != nil {
		return err
	}

	return nil
}

const defaultWindowSize = 32768

// Accept reads and processes messages on a ServerConn. It must be called
// in order to demultiplex messages to any resulting Channels.
func (s *ServerConn) Accept() (Channel, error) {
	if s.err != nil {
		return nil, s.err
	}

	for {
		packet, err := s.readPacket()
		if err != nil {

			s.lock.Lock()
			s.err = err
			s.lock.Unlock()

			for _, c := range s.channels {
				c.dead = true
				c.handleData(nil)
			}

			return nil, err
		}

		switch packet[0] {
		case msgChannelData:
			if len(packet) < 9 {
				// malformed data packet
				return nil, ParseError{msgChannelData}
			}
			peersId := uint32(packet[1])<<24 | uint32(packet[2])<<16 | uint32(packet[3])<<8 | uint32(packet[4])
			s.lock.Lock()
			c, ok := s.channels[peersId]
			if !ok {
				s.lock.Unlock()
				continue
			}
			if length := int(packet[5])<<24 | int(packet[6])<<16 | int(packet[7])<<8 | int(packet[8]); length > 0 {
				packet = packet[9:]
				c.handleData(packet[:length])
			}
			s.lock.Unlock()
		default:
			switch msg := decode(packet).(type) {
			case *channelOpenMsg:
				c := new(channel)
				c.chanType = msg.ChanType
				c.theirId = msg.PeersId
				c.theirWindow = msg.PeersWindow
				c.maxPacketSize = msg.MaxPacketSize
				c.extraData = msg.TypeSpecificData
				c.myWindow = defaultWindowSize
				c.serverConn = s
				c.cond = sync.NewCond(&c.lock)
				c.pendingData = make([]byte, c.myWindow)

				s.lock.Lock()
				c.myId = s.nextChanId
				s.nextChanId++
				s.channels[c.myId] = c
				s.lock.Unlock()
				return c, nil

			case *channelRequestMsg:
				s.lock.Lock()
				c, ok := s.channels[msg.PeersId]
				if !ok {
					s.lock.Unlock()
					continue
				}
				c.handlePacket(msg)
				s.lock.Unlock()

			case *channelEOFMsg:
				s.lock.Lock()
				c, ok := s.channels[msg.PeersId]
				if !ok {
					s.lock.Unlock()
					continue
				}
				c.handlePacket(msg)
				s.lock.Unlock()

			case *channelCloseMsg:
				s.lock.Lock()
				c, ok := s.channels[msg.PeersId]
				if !ok {
					s.lock.Unlock()
					continue
				}
				c.handlePacket(msg)
				s.lock.Unlock()

			case *globalRequestMsg:
				if msg.WantReply {
					if err := s.writePacket([]byte{msgRequestFailure}); err != nil {
						return nil, err
					}
				}

			case UnexpectedMessageError:
				return nil, msg
			case *disconnectMsg:
				return nil, io.EOF
			default:
				// Unknown message. Ignore.
			}
		}
	}

	panic("unreachable")
}

// A Listener implements a network listener (net.Listener) for SSH connections.
type Listener struct {
	listener net.Listener
	config   *ServerConfig
}

// Accept waits for and returns the next incoming SSH connection.
// The receiver should call Handshake() in another goroutine 
// to avoid blocking the accepter.
func (l *Listener) Accept() (*ServerConn, error) {
	c, err := l.listener.Accept()
	if err != nil {
		return nil, err
	}
	conn := Server(c, l.config)
	return conn, nil
}

// Addr returns the listener's network address.
func (l *Listener) Addr() net.Addr {
	return l.listener.Addr()
}

// Close closes the listener.
func (l *Listener) Close() error {
	return l.listener.Close()
}

// Listen creates an SSH listener accepting connections on
// the given network address using net.Listen.
func Listen(network, addr string, config *ServerConfig) (*Listener, error) {
	l, err := net.Listen(network, addr)
	if err != nil {
		return nil, err
	}
	return &Listener{
		l,
		config,
	}, nil
}
