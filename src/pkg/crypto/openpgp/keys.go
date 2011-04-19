// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package openpgp

import (
	"crypto/openpgp/armor"
	"crypto/openpgp/error"
	"crypto/openpgp/packet"
	"io"
	"os"
)

// PublicKeyType is the armor type for a PGP public key.
var PublicKeyType = "PGP PUBLIC KEY BLOCK"
// PrivateKeyType is the armor type for a PGP private key.
var PrivateKeyType = "PGP PRIVATE KEY BLOCK"

// An Entity represents the components of an OpenPGP key: a primary public key
// (which must be a signing key), one or more identities claimed by that key,
// and zero or more subkeys, which may be encryption keys.
type Entity struct {
	PrimaryKey *packet.PublicKey
	PrivateKey *packet.PrivateKey
	Identities map[string]*Identity // indexed by Identity.Name
	Subkeys    []Subkey
}

// An Identity represents an identity claimed by an Entity and zero or more
// assertions by other entities about that claim.
type Identity struct {
	Name          string // by convention, has the form "Full Name (comment) <email@example.com>"
	UserId        *packet.UserId
	SelfSignature *packet.Signature
	Signatures    []*packet.Signature
}

// A Subkey is an additional public key in an Entity. Subkeys can be used for
// encryption.
type Subkey struct {
	PublicKey  *packet.PublicKey
	PrivateKey *packet.PrivateKey
	Sig        *packet.Signature
}

// A Key identifies a specific public key in an Entity. This is either the
// Entity's primary key or a subkey.
type Key struct {
	Entity        *Entity
	PublicKey     *packet.PublicKey
	PrivateKey    *packet.PrivateKey
	SelfSignature *packet.Signature
}

// A KeyRing provides access to public and private keys.
type KeyRing interface {
	// KeysById returns the set of keys that have the given key id.
	KeysById(id uint64) []Key
	// DecryptionKeys returns all private keys that are valid for
	// decryption.
	DecryptionKeys() []Key
}

// An EntityList contains one or more Entities.
type EntityList []*Entity

// KeysById returns the set of keys that have the given key id.
func (el EntityList) KeysById(id uint64) (keys []Key) {
	for _, e := range el {
		if e.PrimaryKey.KeyId == id {
			var selfSig *packet.Signature
			for _, ident := range e.Identities {
				if selfSig == nil {
					selfSig = ident.SelfSignature
				} else if ident.SelfSignature.IsPrimaryId != nil && *ident.SelfSignature.IsPrimaryId {
					selfSig = ident.SelfSignature
					break
				}
			}
			keys = append(keys, Key{e, e.PrimaryKey, e.PrivateKey, selfSig})
		}

		for _, subKey := range e.Subkeys {
			if subKey.PublicKey.KeyId == id {
				keys = append(keys, Key{e, subKey.PublicKey, subKey.PrivateKey, subKey.Sig})
			}
		}
	}
	return
}

// DecryptionKeys returns all private keys that are valid for decryption.
func (el EntityList) DecryptionKeys() (keys []Key) {
	for _, e := range el {
		for _, subKey := range e.Subkeys {
			if subKey.PrivateKey != nil && (!subKey.Sig.FlagsValid || subKey.Sig.FlagEncryptStorage || subKey.Sig.FlagEncryptCommunications) {
				keys = append(keys, Key{e, subKey.PublicKey, subKey.PrivateKey, subKey.Sig})
			}
		}
	}
	return
}

// ReadArmoredKeyRing reads one or more public/private keys from an armor keyring file.
func ReadArmoredKeyRing(r io.Reader) (EntityList, os.Error) {
	block, err := armor.Decode(r)
	if err == os.EOF {
		return nil, error.InvalidArgumentError("no armored data found")
	}
	if err != nil {
		return nil, err
	}
	if block.Type != PublicKeyType && block.Type != PrivateKeyType {
		return nil, error.InvalidArgumentError("expected public or private key block, got: " + block.Type)
	}

	return ReadKeyRing(block.Body)
}

// ReadKeyRing reads one or more public/private keys. Unsupported keys are
// ignored as long as at least a single valid key is found.
func ReadKeyRing(r io.Reader) (el EntityList, err os.Error) {
	packets := packet.NewReader(r)
	var lastUnsupportedError os.Error

	for {
		var e *Entity
		e, err = readEntity(packets)
		if err != nil {
			if _, ok := err.(error.UnsupportedError); ok {
				lastUnsupportedError = err
				err = readToNextPublicKey(packets)
			}
			if err == os.EOF {
				err = nil
				break
			}
			if err != nil {
				el = nil
				break
			}
		} else {
			el = append(el, e)
		}
	}

	if len(el) == 0 && err == nil {
		err = lastUnsupportedError
	}
	return
}

// readToNextPublicKey reads packets until the start of the entity and leaves
// the first packet of the new entity in the Reader.
func readToNextPublicKey(packets *packet.Reader) (err os.Error) {
	var p packet.Packet
	for {
		p, err = packets.Next()
		if err == os.EOF {
			return
		} else if err != nil {
			if _, ok := err.(error.UnsupportedError); ok {
				err = nil
				continue
			}
			return
		}

		if pk, ok := p.(*packet.PublicKey); ok && !pk.IsSubkey {
			packets.Unread(p)
			return
		}
	}

	panic("unreachable")
}

// readEntity reads an entity (public key, identities, subkeys etc) from the
// given Reader.
func readEntity(packets *packet.Reader) (*Entity, os.Error) {
	e := new(Entity)
	e.Identities = make(map[string]*Identity)

	p, err := packets.Next()
	if err != nil {
		return nil, err
	}

	var ok bool
	if e.PrimaryKey, ok = p.(*packet.PublicKey); !ok {
		if e.PrivateKey, ok = p.(*packet.PrivateKey); !ok {
			packets.Unread(p)
			return nil, error.StructuralError("first packet was not a public/private key")
		} else {
			e.PrimaryKey = &e.PrivateKey.PublicKey
		}
	}

	var current *Identity
EachPacket:
	for {
		p, err := packets.Next()
		if err == os.EOF {
			break
		} else if err != nil {
			return nil, err
		}

		switch pkt := p.(type) {
		case *packet.UserId:
			current = new(Identity)
			current.Name = pkt.Id
			current.UserId = pkt
			e.Identities[pkt.Id] = current

			for {
				p, err = packets.Next()
				if err == os.EOF {
					return nil, io.ErrUnexpectedEOF
				} else if err != nil {
					return nil, err
				}

				sig, ok := p.(*packet.Signature)
				if !ok {
					return nil, error.StructuralError("user ID packet not followed by self-signature")
				}

				if sig.SigType == packet.SigTypePositiveCert && sig.IssuerKeyId != nil && *sig.IssuerKeyId == e.PrimaryKey.KeyId {
					if err = e.PrimaryKey.VerifyUserIdSignature(pkt.Id, sig); err != nil {
						return nil, error.StructuralError("user ID self-signature invalid: " + err.String())
					}
					current.SelfSignature = sig
					break
				}
				current.Signatures = append(current.Signatures, sig)
			}
		case *packet.Signature:
			if current == nil {
				return nil, error.StructuralError("signature packet found before user id packet")
			}
			current.Signatures = append(current.Signatures, pkt)
		case *packet.PrivateKey:
			if pkt.IsSubkey == false {
				packets.Unread(p)
				break EachPacket
			}
			err = addSubkey(e, packets, &pkt.PublicKey, pkt)
			if err != nil {
				return nil, err
			}
		case *packet.PublicKey:
			if pkt.IsSubkey == false {
				packets.Unread(p)
				break EachPacket
			}
			err = addSubkey(e, packets, pkt, nil)
			if err != nil {
				return nil, err
			}
		default:
			// we ignore unknown packets
		}
	}

	if len(e.Identities) == 0 {
		return nil, error.StructuralError("entity without any identities")
	}

	return e, nil
}

func addSubkey(e *Entity, packets *packet.Reader, pub *packet.PublicKey, priv *packet.PrivateKey) os.Error {
	var subKey Subkey
	subKey.PublicKey = pub
	subKey.PrivateKey = priv
	p, err := packets.Next()
	if err == os.EOF {
		return io.ErrUnexpectedEOF
	}
	if err != nil {
		return error.StructuralError("subkey signature invalid: " + err.String())
	}
	var ok bool
	subKey.Sig, ok = p.(*packet.Signature)
	if !ok {
		return error.StructuralError("subkey packet not followed by signature")
	}
	if subKey.Sig.SigType != packet.SigTypeSubkeyBinding {
		return error.StructuralError("subkey signature with wrong type")
	}
	err = e.PrimaryKey.VerifyKeySignature(subKey.PublicKey, subKey.Sig)
	if err != nil {
		return error.StructuralError("subkey signature invalid: " + err.String())
	}
	e.Subkeys = append(e.Subkeys, subKey)
	return nil
}
