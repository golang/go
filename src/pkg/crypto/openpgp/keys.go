// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package openpgp

import (
	"crypto"
	"crypto/openpgp/armor"
	"crypto/openpgp/error"
	"crypto/openpgp/packet"
	"crypto/rsa"
	"io"
	"os"
	"time"
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

// primaryIdentity returns the Identity marked as primary or the first identity
// if none are so marked.
func (e *Entity) primaryIdentity() *Identity {
	var firstIdentity *Identity
	for _, ident := range e.Identities {
		if firstIdentity == nil {
			firstIdentity = ident
		}
		if ident.SelfSignature.IsPrimaryId != nil && *ident.SelfSignature.IsPrimaryId {
			return ident
		}
	}
	return firstIdentity
}

// encryptionKey returns the best candidate Key for encrypting a message to the
// given Entity.
func (e *Entity) encryptionKey() Key {
	candidateSubkey := -1

	for i, subkey := range e.Subkeys {
		if subkey.Sig.FlagsValid && subkey.Sig.FlagEncryptCommunications && subkey.PublicKey.PubKeyAlgo.CanEncrypt() {
			candidateSubkey = i
			break
		}
	}

	i := e.primaryIdentity()

	if e.PrimaryKey.PubKeyAlgo.CanEncrypt() {
		// If we don't have any candidate subkeys for encryption and
		// the primary key doesn't have any usage metadata then we
		// assume that the primary key is ok. Or, if the primary key is
		// marked as ok to encrypt to, then we can obviously use it.
		if candidateSubkey == -1 && !i.SelfSignature.FlagsValid || i.SelfSignature.FlagEncryptCommunications && i.SelfSignature.FlagsValid {
			return Key{e, e.PrimaryKey, e.PrivateKey, i.SelfSignature}
		}
	}

	if candidateSubkey != -1 {
		subkey := e.Subkeys[candidateSubkey]
		return Key{e, subkey.PublicKey, subkey.PrivateKey, subkey.Sig}
	}

	// This Entity appears to be signing only.
	return Key{}
}

// signingKey return the best candidate Key for signing a message with this
// Entity.
func (e *Entity) signingKey() Key {
	candidateSubkey := -1

	for i, subkey := range e.Subkeys {
		if subkey.Sig.FlagsValid && subkey.Sig.FlagSign && subkey.PublicKey.PubKeyAlgo.CanSign() {
			candidateSubkey = i
			break
		}
	}

	i := e.primaryIdentity()

	// If we have no candidate subkey then we assume that it's ok to sign
	// with the primary key.
	if candidateSubkey == -1 || i.SelfSignature.FlagsValid && i.SelfSignature.FlagSign {
		return Key{e, e.PrimaryKey, e.PrivateKey, i.SelfSignature}
	}

	subkey := e.Subkeys[candidateSubkey]
	return Key{e, subkey.PublicKey, subkey.PrivateKey, subkey.Sig}
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

	if !e.PrimaryKey.PubKeyAlgo.CanSign() {
		return nil, error.StructuralError("primary key cannot be used for signatures")
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

				if (sig.SigType == packet.SigTypePositiveCert || sig.SigType == packet.SigTypeGenericCert) && sig.IssuerKeyId != nil && *sig.IssuerKeyId == e.PrimaryKey.KeyId {
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

const defaultRSAKeyBits = 2048

// NewEntity returns an Entity that contains a fresh RSA/RSA keypair with a
// single identity composed of the given full name, comment and email, any of
// which may be empty but must not contain any of "()<>\x00".
func NewEntity(rand io.Reader, currentTimeSecs int64, name, comment, email string) (*Entity, os.Error) {
	uid := packet.NewUserId(name, comment, email)
	if uid == nil {
		return nil, error.InvalidArgumentError("user id field contained invalid characters")
	}
	signingPriv, err := rsa.GenerateKey(rand, defaultRSAKeyBits)
	if err != nil {
		return nil, err
	}
	encryptingPriv, err := rsa.GenerateKey(rand, defaultRSAKeyBits)
	if err != nil {
		return nil, err
	}

	t := uint32(currentTimeSecs)

	e := &Entity{
		PrimaryKey: packet.NewRSAPublicKey(t, &signingPriv.PublicKey, false /* not a subkey */ ),
		PrivateKey: packet.NewRSAPrivateKey(t, signingPriv, false /* not a subkey */ ),
		Identities: make(map[string]*Identity),
	}
	isPrimaryId := true
	e.Identities[uid.Id] = &Identity{
		Name:   uid.Name,
		UserId: uid,
		SelfSignature: &packet.Signature{
			CreationTime: t,
			SigType:      packet.SigTypePositiveCert,
			PubKeyAlgo:   packet.PubKeyAlgoRSA,
			Hash:         crypto.SHA256,
			IsPrimaryId:  &isPrimaryId,
			FlagsValid:   true,
			FlagSign:     true,
			FlagCertify:  true,
			IssuerKeyId:  &e.PrimaryKey.KeyId,
		},
	}

	e.Subkeys = make([]Subkey, 1)
	e.Subkeys[0] = Subkey{
		PublicKey:  packet.NewRSAPublicKey(t, &encryptingPriv.PublicKey, true /* is a subkey */ ),
		PrivateKey: packet.NewRSAPrivateKey(t, encryptingPriv, true /* is a subkey */ ),
		Sig: &packet.Signature{
			CreationTime:              t,
			SigType:                   packet.SigTypeSubkeyBinding,
			PubKeyAlgo:                packet.PubKeyAlgoRSA,
			Hash:                      crypto.SHA256,
			FlagsValid:                true,
			FlagEncryptStorage:        true,
			FlagEncryptCommunications: true,
			IssuerKeyId:               &e.PrimaryKey.KeyId,
		},
	}

	return e, nil
}

// SerializePrivate serializes an Entity, including private key material, to
// the given Writer. For now, it must only be used on an Entity returned from
// NewEntity.
func (e *Entity) SerializePrivate(w io.Writer) (err os.Error) {
	err = e.PrivateKey.Serialize(w)
	if err != nil {
		return
	}
	for _, ident := range e.Identities {
		err = ident.UserId.Serialize(w)
		if err != nil {
			return
		}
		err = ident.SelfSignature.SignUserId(ident.UserId.Id, e.PrimaryKey, e.PrivateKey)
		if err != nil {
			return
		}
		err = ident.SelfSignature.Serialize(w)
		if err != nil {
			return
		}
	}
	for _, subkey := range e.Subkeys {
		err = subkey.PrivateKey.Serialize(w)
		if err != nil {
			return
		}
		err = subkey.Sig.SignKey(subkey.PublicKey, e.PrivateKey)
		if err != nil {
			return
		}
		err = subkey.Sig.Serialize(w)
		if err != nil {
			return
		}
	}
	return nil
}

// Serialize writes the public part of the given Entity to w. (No private
// key material will be output).
func (e *Entity) Serialize(w io.Writer) os.Error {
	err := e.PrimaryKey.Serialize(w)
	if err != nil {
		return err
	}
	for _, ident := range e.Identities {
		err = ident.UserId.Serialize(w)
		if err != nil {
			return err
		}
		err = ident.SelfSignature.Serialize(w)
		if err != nil {
			return err
		}
		for _, sig := range ident.Signatures {
			err = sig.Serialize(w)
			if err != nil {
				return err
			}
		}
	}
	for _, subkey := range e.Subkeys {
		err = subkey.PublicKey.Serialize(w)
		if err != nil {
			return err
		}
		err = subkey.Sig.Serialize(w)
		if err != nil {
			return err
		}
	}
	return nil
}

// SignIdentity adds a signature to e, from signer, attesting that identity is
// associated with e. The provided identity must already be an element of
// e.Identities and the private key of signer must have been decrypted if
// necessary.
func (e *Entity) SignIdentity(identity string, signer *Entity) os.Error {
	if signer.PrivateKey == nil {
		return error.InvalidArgumentError("signing Entity must have a private key")
	}
	if signer.PrivateKey.Encrypted {
		return error.InvalidArgumentError("signing Entity's private key must be decrypted")
	}
	ident, ok := e.Identities[identity]
	if !ok {
		return error.InvalidArgumentError("given identity string not found in Entity")
	}

	sig := &packet.Signature{
		SigType:      packet.SigTypeGenericCert,
		PubKeyAlgo:   signer.PrivateKey.PubKeyAlgo,
		Hash:         crypto.SHA256,
		CreationTime: uint32(time.Seconds()),
		IssuerKeyId:  &signer.PrivateKey.KeyId,
	}
	if err := sig.SignKey(e.PrimaryKey, signer.PrivateKey); err != nil {
		return err
	}
	ident.Signatures = append(ident.Signatures, sig)
	return nil
}
