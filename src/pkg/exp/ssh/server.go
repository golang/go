// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"big"
	"bytes"
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	_ "crypto/sha1"
	"crypto/x509"
	"encoding/pem"
	"net"
	"os"
	"sync"
)

// Server represents an SSH server. A Server may have several ServerConnections.
type Server struct {
	rsa           *rsa.PrivateKey
	rsaSerialized []byte

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

// SetRSAPrivateKey sets the private key for a Server. A Server must have a
// private key configured in order to accept connections. The private key must
// be in the form of a PEM encoded, PKCS#1, RSA private key. The file "id_rsa"
// typically contains such a key.
func (s *Server) SetRSAPrivateKey(pemBytes []byte) os.Error {
	block, _ := pem.Decode(pemBytes)
	if block == nil {
		return os.NewError("ssh: no key found")
	}
	var err os.Error
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
// acceptable for a user. The cache only applies to a single ServerConnection.
type cachedPubKey struct {
	user, algo string
	pubKey     []byte
	result     bool
}

const maxCachedPubKeys = 16

// ServerConnection represents an incomming connection to a Server.
type ServerConnection struct {
	Server *Server

	*transport

	channels   map[uint32]*channel
	nextChanId uint32

	// lock protects err and also allows Channels to serialise their writes
	// to out.
	lock sync.RWMutex
	err  os.Error

	// cachedPubKeys contains the cache results of tests for public keys.
	// Since SSH clients will query whether a public key is acceptable
	// before attempting to authenticate with it, we end up with duplicate
	// queries for public key validity.
	cachedPubKeys []cachedPubKey
}

// kexDH performs Diffie-Hellman key agreement on a ServerConnection. The
// returned values are given the same names as in RFC 4253, section 8.
func (s *ServerConnection) kexDH(group *dhGroup, hashFunc crypto.Hash, magics *handshakeMagics, hostKeyAlgo string) (H, K []byte, err os.Error) {
	packet, err := s.readPacket()
	if err != nil {
		return
	}
	var kexDHInit kexDHInitMsg
	if err = unmarshal(&kexDHInit, packet, msgKexDHInit); err != nil {
		return
	}

	if kexDHInit.X.Sign() == 0 || kexDHInit.X.Cmp(group.p) >= 0 {
		return nil, nil, os.NewError("client DH parameter out of bounds")
	}

	y, err := rand.Int(rand.Reader, group.p)
	if err != nil {
		return
	}

	Y := new(big.Int).Exp(group.g, y, group.p)
	kInt := new(big.Int).Exp(kexDHInit.X, y, group.p)

	var serializedHostKey []byte
	switch hostKeyAlgo {
	case hostAlgoRSA:
		serializedHostKey = s.Server.rsaSerialized
	default:
		return nil, nil, os.NewError("internal error")
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
		sig, err = rsa.SignPKCS1v15(rand.Reader, s.Server.rsa, hashFunc, hh)
		if err != nil {
			return
		}
	default:
		return nil, nil, os.NewError("internal error")
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

// Handshake performs an SSH transport and client authentication on the given ServerConnection.
func (s *ServerConnection) Handshake(conn net.Conn) os.Error {
	var magics handshakeMagics
	s.transport = newTransport(conn, rand.Reader)

	if _, err := conn.Write(serverVersion); err != nil {
		return err
	}
	magics.serverVersion = serverVersion[:len(serverVersion)-2]

	version, ok := readVersion(s.transport)
	if !ok {
		return os.NewError("failed to read version string from client")
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
		return os.NewError("ssh: no common algorithms")
	}

	if clientKexInit.FirstKexFollows && kexAlgo != clientKexInit.KexAlgos[0] {
		// The client sent a Kex message for the wrong algorithm,
		// which we have to ignore.
		_, err := s.readPacket()
		if err != nil {
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
		err = os.NewError("ssh: internal error")
	}

	if err != nil {
		return err
	}

	packet = []byte{msgNewKeys}
	if err = s.writePacket(packet); err != nil {
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

	packet, err = s.readPacket()
	if err != nil {
		return err
	}

	var serviceRequest serviceRequestMsg
	if err = unmarshal(&serviceRequest, packet, msgServiceRequest); err != nil {
		return err
	}
	if serviceRequest.Service != serviceUserAuth {
		return os.NewError("ssh: requested service '" + serviceRequest.Service + "' before authenticating")
	}

	serviceAccept := serviceAcceptMsg{
		Service: serviceUserAuth,
	}
	packet = marshal(msgServiceAccept, serviceAccept)
	if err = s.writePacket(packet); err != nil {
		return err
	}

	if err = s.authenticate(H); err != nil {
		return err
	}

	s.channels = make(map[uint32]*channel)
	return nil
}

func isAcceptableAlgo(algo string) bool {
	return algo == hostAlgoRSA
}

// testPubKey returns true if the given public key is acceptable for the user.
func (s *ServerConnection) testPubKey(user, algo string, pubKey []byte) bool {
	if s.Server.PubKeyCallback == nil || !isAcceptableAlgo(algo) {
		return false
	}

	for _, c := range s.cachedPubKeys {
		if c.user == user && c.algo == algo && bytes.Equal(c.pubKey, pubKey) {
			return c.result
		}
	}

	result := s.Server.PubKeyCallback(user, algo, pubKey)
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

func (s *ServerConnection) authenticate(H []byte) os.Error {
	var userAuthReq userAuthRequestMsg
	var err os.Error
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
			return os.NewError("ssh: client attempted to negotiate for unknown service: " + userAuthReq.Service)
		}

		switch userAuthReq.Method {
		case "none":
			if s.Server.NoClientAuth {
				break userAuthLoop
			}
		case "password":
			if s.Server.PasswordCallback == nil {
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

			if s.Server.PasswordCallback(userAuthReq.User, string(password)) {
				break userAuthLoop
			}
		case "publickey":
			if s.Server.PubKeyCallback == nil {
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
					return os.NewError("ssh: isAcceptableAlgo incorrect")
				}
				if s.testPubKey(userAuthReq.User, algo, pubKey) {
					break userAuthLoop
				}
			}
		}

		var failureMsg userAuthFailureMsg
		if s.Server.PasswordCallback != nil {
			failureMsg.Methods = append(failureMsg.Methods, "password")
		}
		if s.Server.PubKeyCallback != nil {
			failureMsg.Methods = append(failureMsg.Methods, "publickey")
		}

		if len(failureMsg.Methods) == 0 {
			return os.NewError("ssh: no authentication methods configured but NoClientAuth is also false")
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

// Accept reads and processes messages on a ServerConnection. It must be called
// in order to demultiplex messages to any resulting Channels.
func (s *ServerConnection) Accept() (Channel, os.Error) {
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
				continue
			}
			c.handlePacket(msg)
			s.lock.Unlock()

		case *channelData:
			s.lock.Lock()
			c, ok := s.channels[msg.PeersId]
			if !ok {
				continue
			}
			c.handleData(msg.Payload)
			s.lock.Unlock()

		case *channelEOFMsg:
			s.lock.Lock()
			c, ok := s.channels[msg.PeersId]
			if !ok {
				continue
			}
			c.handlePacket(msg)
			s.lock.Unlock()

		case *channelCloseMsg:
			s.lock.Lock()
			c, ok := s.channels[msg.PeersId]
			if !ok {
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
			return nil, os.EOF
		default:
			// Unknown message. Ignore.
		}
	}

	panic("unreachable")
}
