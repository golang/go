// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"errors"
	"io"
)

// authenticate authenticates with the remote server. See RFC 4252.
func (c *ClientConn) authenticate(session []byte) error {
	// initiate user auth session
	if err := c.writePacket(marshal(msgServiceRequest, serviceRequestMsg{serviceUserAuth})); err != nil {
		return err
	}
	packet, err := c.readPacket()
	if err != nil {
		return err
	}
	var serviceAccept serviceAcceptMsg
	if err := unmarshal(&serviceAccept, packet, msgServiceAccept); err != nil {
		return err
	}
	// during the authentication phase the client first attempts the "none" method
	// then any untried methods suggested by the server.
	tried, remain := make(map[string]bool), make(map[string]bool)
	for auth := ClientAuth(new(noneAuth)); auth != nil; {
		ok, methods, err := auth.auth(session, c.config.User, c.transport, c.config.rand())
		if err != nil {
			return err
		}
		if ok {
			// success
			return nil
		}
		tried[auth.method()] = true
		delete(remain, auth.method())
		for _, meth := range methods {
			if tried[meth] {
				// if we've tried meth already, skip it.
				continue
			}
			remain[meth] = true
		}
		auth = nil
		for _, a := range c.config.Auth {
			if remain[a.method()] {
				auth = a
				break
			}
		}
	}
	return errors.New("ssh: unable to authenticate, no supported methods remain")
}

// A ClientAuth represents an instance of an RFC 4252 authentication method.
type ClientAuth interface {
	// auth authenticates user over transport t.
	// Returns true if authentication is successful.
	// If authentication is not successful, a []string of alternative
	// method names is returned.
	auth(session []byte, user string, t *transport, rand io.Reader) (bool, []string, error)

	// method returns the RFC 4252 method name.
	method() string
}

// "none" authentication, RFC 4252 section 5.2.
type noneAuth int

func (n *noneAuth) auth(session []byte, user string, t *transport, rand io.Reader) (bool, []string, error) {
	if err := t.writePacket(marshal(msgUserAuthRequest, userAuthRequestMsg{
		User:    user,
		Service: serviceSSH,
		Method:  "none",
	})); err != nil {
		return false, nil, err
	}

	return handleAuthResponse(t)
}

func (n *noneAuth) method() string {
	return "none"
}

// "password" authentication, RFC 4252 Section 8.
type passwordAuth struct {
	ClientPassword
}

func (p *passwordAuth) auth(session []byte, user string, t *transport, rand io.Reader) (bool, []string, error) {
	type passwordAuthMsg struct {
		User     string
		Service  string
		Method   string
		Reply    bool
		Password string
	}

	pw, err := p.Password(user)
	if err != nil {
		return false, nil, err
	}

	if err := t.writePacket(marshal(msgUserAuthRequest, passwordAuthMsg{
		User:     user,
		Service:  serviceSSH,
		Method:   "password",
		Reply:    false,
		Password: pw,
	})); err != nil {
		return false, nil, err
	}

	return handleAuthResponse(t)
}

func (p *passwordAuth) method() string {
	return "password"
}

// A ClientPassword implements access to a client's passwords.
type ClientPassword interface {
	// Password returns the password to use for user.
	Password(user string) (password string, err error)
}

// ClientAuthPassword returns a ClientAuth using password authentication.
func ClientAuthPassword(impl ClientPassword) ClientAuth {
	return &passwordAuth{impl}
}

// ClientKeyring implements access to a client key ring.
type ClientKeyring interface {
	// Key returns the i'th rsa.Publickey or dsa.Publickey, or nil if
	// no key exists at i.
	Key(i int) (key interface{}, err error)

	// Sign returns a signature of the given data using the i'th key
	// and the supplied random source.
	Sign(i int, rand io.Reader, data []byte) (sig []byte, err error)
}

// "publickey" authentication, RFC 4252 Section 7.
type publickeyAuth struct {
	ClientKeyring
}

type publickeyAuthMsg struct {
	User    string
	Service string
	Method  string
	// HasSig indicates to the reciver packet that the auth request is signed and
	// should be used for authentication of the request.
	HasSig   bool
	Algoname string
	Pubkey   string
	// Sig is defined as []byte so marshal will exclude it during validateKey
	Sig []byte `ssh:"rest"`
}

func (p *publickeyAuth) auth(session []byte, user string, t *transport, rand io.Reader) (bool, []string, error) {

	// Authentication is performed in two stages. The first stage sends an
	// enquiry to test if each key is acceptable to the remote. The second
	// stage attempts to authenticate with the valid keys obtained in the
	// first stage.

	var index int
	// a map of public keys to their index in the keyring
	validKeys := make(map[int]interface{})
	for {
		key, err := p.Key(index)
		if err != nil {
			return false, nil, err
		}
		if key == nil {
			// no more keys in the keyring
			break
		}

		if ok, err := p.validateKey(key, user, t); ok {
			validKeys[index] = key
		} else {
			if err != nil {
				return false, nil, err
			}
		}
		index++
	}

	// methods that may continue if this auth is not successful.
	var methods []string
	for i, key := range validKeys {
		pubkey := serializePublickey(key)
		algoname := algoName(key)
		sign, err := p.Sign(i, rand, buildDataSignedForAuth(session, userAuthRequestMsg{
			User:    user,
			Service: serviceSSH,
			Method:  p.method(),
		}, []byte(algoname), pubkey))
		if err != nil {
			return false, nil, err
		}
		// manually wrap the serialized signature in a string
		s := serializeSignature(algoname, sign)
		sig := make([]byte, stringLength(s))
		marshalString(sig, s)
		msg := publickeyAuthMsg{
			User:     user,
			Service:  serviceSSH,
			Method:   p.method(),
			HasSig:   true,
			Algoname: algoname,
			Pubkey:   string(pubkey),
			Sig:      sig,
		}
		p := marshal(msgUserAuthRequest, msg)
		if err := t.writePacket(p); err != nil {
			return false, nil, err
		}
		success, methods, err := handleAuthResponse(t)
		if err != nil {
			return false, nil, err
		}
		if success {
			return success, methods, err
		}
	}
	return false, methods, nil
}

// validateKey validates the key provided it is acceptable to the server.
func (p *publickeyAuth) validateKey(key interface{}, user string, t *transport) (bool, error) {
	pubkey := serializePublickey(key)
	algoname := algoName(key)
	msg := publickeyAuthMsg{
		User:     user,
		Service:  serviceSSH,
		Method:   p.method(),
		HasSig:   false,
		Algoname: algoname,
		Pubkey:   string(pubkey),
	}
	if err := t.writePacket(marshal(msgUserAuthRequest, msg)); err != nil {
		return false, err
	}

	return p.confirmKeyAck(key, t)
}

func (p *publickeyAuth) confirmKeyAck(key interface{}, t *transport) (bool, error) {
	pubkey := serializePublickey(key)
	algoname := algoName(key)

	for {
		packet, err := t.readPacket()
		if err != nil {
			return false, err
		}
		switch packet[0] {
		case msgUserAuthBanner:
			// TODO(gpaul): add callback to present the banner to the user
		case msgUserAuthPubKeyOk:
			msg := decode(packet).(*userAuthPubKeyOkMsg)
			if msg.Algo != algoname || msg.PubKey != string(pubkey) {
				return false, nil
			}
			return true, nil
		case msgUserAuthFailure:
			return false, nil
		default:
			return false, UnexpectedMessageError{msgUserAuthSuccess, packet[0]}
		}
	}
	panic("unreachable")
}

func (p *publickeyAuth) method() string {
	return "publickey"
}

// ClientAuthKeyring returns a ClientAuth using public key authentication.
func ClientAuthKeyring(impl ClientKeyring) ClientAuth {
	return &publickeyAuth{impl}
}

// handleAuthResponse returns whether the preceding authentication request succeeded
// along with a list of remaining authentication methods to try next and
// an error if an unexpected response was received.
func handleAuthResponse(t *transport) (bool, []string, error) {
	for {
		packet, err := t.readPacket()
		if err != nil {
			return false, nil, err
		}

		switch packet[0] {
		case msgUserAuthBanner:
			// TODO: add callback to present the banner to the user
		case msgUserAuthFailure:
			msg := decode(packet).(*userAuthFailureMsg)
			return false, msg.Methods, nil
		case msgUserAuthSuccess:
			return true, nil, nil
		case msgDisconnect:
			return false, nil, io.EOF
		default:
			return false, nil, UnexpectedMessageError{msgUserAuthSuccess, packet[0]}
		}
	}
	panic("unreachable")
}
