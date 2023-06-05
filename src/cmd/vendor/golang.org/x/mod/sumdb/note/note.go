// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package note defines the notes signed by the Go module database server.
//
// A note is text signed by one or more server keys.
// The text should be ignored unless the note is signed by
// a trusted server key and the signature has been verified
// using the server's public key.
//
// A server's public key is identified by a name, typically the "host[/path]"
// giving the base URL of the server's transparency log.
// The syntactic restrictions on a name are that it be non-empty,
// well-formed UTF-8 containing neither Unicode spaces nor plus (U+002B).
//
// A Go module database server signs texts using public key cryptography.
// A given server may have multiple public keys, each
// identified by a 32-bit hash of the public key.
//
// # Verifying Notes
//
// A Verifier allows verification of signatures by one server public key.
// It can report the name of the server and the uint32 hash of the key,
// and it can verify a purported signature by that key.
//
// The standard implementation of a Verifier is constructed
// by NewVerifier starting from a verifier key, which is a
// plain text string of the form "<name>+<hash>+<keydata>".
//
// A Verifiers allows looking up a Verifier by the combination
// of server name and key hash.
//
// The standard implementation of a Verifiers is constructed
// by VerifierList from a list of known verifiers.
//
// A Note represents a text with one or more signatures.
// An implementation can reject a note with too many signatures
// (for example, more than 100 signatures).
//
// A Signature represents a signature on a note, verified or not.
//
// The Open function takes as input a signed message
// and a set of known verifiers. It decodes and verifies
// the message signatures and returns a Note structure
// containing the message text and (verified or unverified) signatures.
//
// # Signing Notes
//
// A Signer allows signing a text with a given key.
// It can report the name of the server and the hash of the key
// and can sign a raw text using that key.
//
// The standard implementation of a Signer is constructed
// by NewSigner starting from an encoded signer key, which is a
// plain text string of the form "PRIVATE+KEY+<name>+<hash>+<keydata>".
// Anyone with an encoded signer key can sign messages using that key,
// so it must be kept secret. The encoding begins with the literal text
// "PRIVATE+KEY" to avoid confusion with the public server key.
//
// The Sign function takes as input a Note and a list of Signers
// and returns an encoded, signed message.
//
// # Signed Note Format
//
// A signed note consists of a text ending in newline (U+000A),
// followed by a blank line (only a newline),
// followed by one or more signature lines of this form:
// em dash (U+2014), space (U+0020),
// server name, space, base64-encoded signature, newline.
//
// Signed notes must be valid UTF-8 and must not contain any
// ASCII control characters (those below U+0020) other than newline.
//
// A signature is a base64 encoding of 4+n bytes.
//
// The first four bytes in the signature are the uint32 key hash
// stored in big-endian order.
//
// The remaining n bytes are the result of using the specified key
// to sign the note text (including the final newline but not the
// separating blank line).
//
// # Generating Keys
//
// There is only one key type, Ed25519 with algorithm identifier 1.
// New key types may be introduced in the future as needed,
// although doing so will require deploying the new algorithms to all clients
// before starting to depend on them for signatures.
//
// The GenerateKey function generates and returns a new signer
// and corresponding verifier.
//
// # Example
//
// Here is a well-formed signed note:
//
//	If you think cryptography is the answer to your problem,
//	then you don't know what your problem is.
//
//	— PeterNeumann x08go/ZJkuBS9UG/SffcvIAQxVBtiFupLLr8pAcElZInNIuGUgYN1FFYC2pZSNXgKvqfqdngotpRZb6KE6RyyBwJnAM=
//
// It can be constructed and displayed using:
//
//	skey := "PRIVATE+KEY+PeterNeumann+c74f20a3+AYEKFALVFGyNhPJEMzD1QIDr+Y7hfZx09iUvxdXHKDFz"
//	text := "If you think cryptography is the answer to your problem,\n" +
//		"then you don't know what your problem is.\n"
//
//	signer, err := note.NewSigner(skey)
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	msg, err := note.Sign(&note.Note{Text: text}, signer)
//	if err != nil {
//		log.Fatal(err)
//	}
//	os.Stdout.Write(msg)
//
// The note's text is two lines, including the final newline,
// and the text is purportedly signed by a server named
// "PeterNeumann". (Although server names are canonically
// base URLs, the only syntactic requirement is that they
// not contain spaces or newlines).
//
// If Open is given access to a Verifiers including the
// Verifier for this key, then it will succeed at verifying
// the encoded message and returning the parsed Note:
//
//	vkey := "PeterNeumann+c74f20a3+ARpc2QcUPDhMQegwxbzhKqiBfsVkmqq/LDE4izWy10TW"
//	msg := []byte("If you think cryptography is the answer to your problem,\n" +
//		"then you don't know what your problem is.\n" +
//		"\n" +
//		"— PeterNeumann x08go/ZJkuBS9UG/SffcvIAQxVBtiFupLLr8pAcElZInNIuGUgYN1FFYC2pZSNXgKvqfqdngotpRZb6KE6RyyBwJnAM=\n")
//
//	verifier, err := note.NewVerifier(vkey)
//	if err != nil {
//		log.Fatal(err)
//	}
//	verifiers := note.VerifierList(verifier)
//
//	n, err := note.Open([]byte(msg), verifiers)
//	if err != nil {
//		log.Fatal(err)
//	}
//	fmt.Printf("%s (%08x):\n%s", n.Sigs[0].Name, n.Sigs[0].Hash, n.Text)
//
// You can add your own signature to this message by re-signing the note:
//
//	skey, vkey, err := note.GenerateKey(rand.Reader, "EnochRoot")
//	if err != nil {
//		log.Fatal(err)
//	}
//	_ = vkey // give to verifiers
//
//	me, err := note.NewSigner(skey)
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	msg, err := note.Sign(n, me)
//	if err != nil {
//		log.Fatal(err)
//	}
//	os.Stdout.Write(msg)
//
// This will print a doubly-signed message, like:
//
//	If you think cryptography is the answer to your problem,
//	then you don't know what your problem is.
//
//	— PeterNeumann x08go/ZJkuBS9UG/SffcvIAQxVBtiFupLLr8pAcElZInNIuGUgYN1FFYC2pZSNXgKvqfqdngotpRZb6KE6RyyBwJnAM=
//	— EnochRoot rwz+eBzmZa0SO3NbfRGzPCpDckykFXSdeX+MNtCOXm2/5n2tiOHp+vAF1aGrQ5ovTG01oOTGwnWLox33WWd1RvMc+QQ=
package note

import (
	"bytes"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

// A Verifier verifies messages signed with a specific key.
type Verifier interface {
	// Name returns the server name associated with the key.
	Name() string

	// KeyHash returns the key hash.
	KeyHash() uint32

	// Verify reports whether sig is a valid signature of msg.
	Verify(msg, sig []byte) bool
}

// A Signer signs messages using a specific key.
type Signer interface {
	// Name returns the server name associated with the key.
	Name() string

	// KeyHash returns the key hash.
	KeyHash() uint32

	// Sign returns a signature for the given message.
	Sign(msg []byte) ([]byte, error)
}

// keyHash computes the key hash for the given server name and encoded public key.
func keyHash(name string, key []byte) uint32 {
	h := sha256.New()
	h.Write([]byte(name))
	h.Write([]byte("\n"))
	h.Write(key)
	sum := h.Sum(nil)
	return binary.BigEndian.Uint32(sum)
}

var (
	errVerifierID   = errors.New("malformed verifier id")
	errVerifierAlg  = errors.New("unknown verifier algorithm")
	errVerifierHash = errors.New("invalid verifier hash")
)

const (
	algEd25519 = 1
)

// isValidName reports whether name is valid.
// It must be non-empty and not have any Unicode spaces or pluses.
func isValidName(name string) bool {
	return name != "" && utf8.ValidString(name) && strings.IndexFunc(name, unicode.IsSpace) < 0 && !strings.Contains(name, "+")
}

// NewVerifier construct a new Verifier from an encoded verifier key.
func NewVerifier(vkey string) (Verifier, error) {
	name, vkey := chop(vkey, "+")
	hash16, key64 := chop(vkey, "+")
	hash, err1 := strconv.ParseUint(hash16, 16, 32)
	key, err2 := base64.StdEncoding.DecodeString(key64)
	if len(hash16) != 8 || err1 != nil || err2 != nil || !isValidName(name) || len(key) == 0 {
		return nil, errVerifierID
	}
	if uint32(hash) != keyHash(name, key) {
		return nil, errVerifierHash
	}

	v := &verifier{
		name: name,
		hash: uint32(hash),
	}

	alg, key := key[0], key[1:]
	switch alg {
	default:
		return nil, errVerifierAlg

	case algEd25519:
		if len(key) != 32 {
			return nil, errVerifierID
		}
		v.verify = func(msg, sig []byte) bool {
			return ed25519.Verify(key, msg, sig)
		}
	}

	return v, nil
}

// chop chops s at the first instance of sep, if any,
// and returns the text before and after sep.
// If sep is not present, chop returns before is s and after is empty.
func chop(s, sep string) (before, after string) {
	i := strings.Index(s, sep)
	if i < 0 {
		return s, ""
	}
	return s[:i], s[i+len(sep):]
}

// verifier is a trivial Verifier implementation.
type verifier struct {
	name   string
	hash   uint32
	verify func([]byte, []byte) bool
}

func (v *verifier) Name() string                { return v.name }
func (v *verifier) KeyHash() uint32             { return v.hash }
func (v *verifier) Verify(msg, sig []byte) bool { return v.verify(msg, sig) }

// NewSigner constructs a new Signer from an encoded signer key.
func NewSigner(skey string) (Signer, error) {
	priv1, skey := chop(skey, "+")
	priv2, skey := chop(skey, "+")
	name, skey := chop(skey, "+")
	hash16, key64 := chop(skey, "+")
	hash, err1 := strconv.ParseUint(hash16, 16, 32)
	key, err2 := base64.StdEncoding.DecodeString(key64)
	if priv1 != "PRIVATE" || priv2 != "KEY" || len(hash16) != 8 || err1 != nil || err2 != nil || !isValidName(name) || len(key) == 0 {
		return nil, errSignerID
	}

	// Note: hash is the hash of the public key and we have the private key.
	// Must verify hash after deriving public key.

	s := &signer{
		name: name,
		hash: uint32(hash),
	}

	var pubkey []byte

	alg, key := key[0], key[1:]
	switch alg {
	default:
		return nil, errSignerAlg

	case algEd25519:
		if len(key) != 32 {
			return nil, errSignerID
		}
		key = ed25519.NewKeyFromSeed(key)
		pubkey = append([]byte{algEd25519}, key[32:]...)
		s.sign = func(msg []byte) ([]byte, error) {
			return ed25519.Sign(key, msg), nil
		}
	}

	if uint32(hash) != keyHash(name, pubkey) {
		return nil, errSignerHash
	}

	return s, nil
}

var (
	errSignerID   = errors.New("malformed verifier id")
	errSignerAlg  = errors.New("unknown verifier algorithm")
	errSignerHash = errors.New("invalid verifier hash")
)

// signer is a trivial Signer implementation.
type signer struct {
	name string
	hash uint32
	sign func([]byte) ([]byte, error)
}

func (s *signer) Name() string                    { return s.name }
func (s *signer) KeyHash() uint32                 { return s.hash }
func (s *signer) Sign(msg []byte) ([]byte, error) { return s.sign(msg) }

// GenerateKey generates a signer and verifier key pair for a named server.
// The signer key skey is private and must be kept secret.
func GenerateKey(rand io.Reader, name string) (skey, vkey string, err error) {
	pub, priv, err := ed25519.GenerateKey(rand)
	if err != nil {
		return "", "", err
	}
	pubkey := append([]byte{algEd25519}, pub...)
	privkey := append([]byte{algEd25519}, priv.Seed()...)
	h := keyHash(name, pubkey)

	skey = fmt.Sprintf("PRIVATE+KEY+%s+%08x+%s", name, h, base64.StdEncoding.EncodeToString(privkey))
	vkey = fmt.Sprintf("%s+%08x+%s", name, h, base64.StdEncoding.EncodeToString(pubkey))
	return skey, vkey, nil
}

// NewEd25519VerifierKey returns an encoded verifier key using the given name
// and Ed25519 public key.
func NewEd25519VerifierKey(name string, key ed25519.PublicKey) (string, error) {
	if len(key) != ed25519.PublicKeySize {
		return "", fmt.Errorf("invalid public key size %d, expected %d", len(key), ed25519.PublicKeySize)
	}

	pubkey := append([]byte{algEd25519}, key...)
	hash := keyHash(name, pubkey)

	b64Key := base64.StdEncoding.EncodeToString(pubkey)
	return fmt.Sprintf("%s+%08x+%s", name, hash, b64Key), nil
}

// A Verifiers is a collection of known verifier keys.
type Verifiers interface {
	// Verifier returns the Verifier associated with the key
	// identified by the name and hash.
	// If the name, hash pair is unknown, Verifier should return
	// an UnknownVerifierError.
	Verifier(name string, hash uint32) (Verifier, error)
}

// An UnknownVerifierError indicates that the given key is not known.
// The Open function records signatures without associated verifiers as
// unverified signatures.
type UnknownVerifierError struct {
	Name    string
	KeyHash uint32
}

func (e *UnknownVerifierError) Error() string {
	return fmt.Sprintf("unknown key %s+%08x", e.Name, e.KeyHash)
}

// An ambiguousVerifierError indicates that the given name and hash
// match multiple keys passed to VerifierList.
// (If this happens, some malicious actor has taken control of the
// verifier list, at which point we may as well give up entirely,
// but we diagnose the problem instead.)
type ambiguousVerifierError struct {
	name string
	hash uint32
}

func (e *ambiguousVerifierError) Error() string {
	return fmt.Sprintf("ambiguous key %s+%08x", e.name, e.hash)
}

// VerifierList returns a Verifiers implementation that uses the given list of verifiers.
func VerifierList(list ...Verifier) Verifiers {
	m := make(verifierMap)
	for _, v := range list {
		k := nameHash{v.Name(), v.KeyHash()}
		m[k] = append(m[k], v)
	}
	return m
}

type nameHash struct {
	name string
	hash uint32
}

type verifierMap map[nameHash][]Verifier

func (m verifierMap) Verifier(name string, hash uint32) (Verifier, error) {
	v, ok := m[nameHash{name, hash}]
	if !ok {
		return nil, &UnknownVerifierError{name, hash}
	}
	if len(v) > 1 {
		return nil, &ambiguousVerifierError{name, hash}
	}
	return v[0], nil
}

// A Note is a text and signatures.
type Note struct {
	Text           string      // text of note
	Sigs           []Signature // verified signatures
	UnverifiedSigs []Signature // unverified signatures
}

// A Signature is a single signature found in a note.
type Signature struct {
	// Name and Hash give the name and key hash
	// for the key that generated the signature.
	Name string
	Hash uint32

	// Base64 records the base64-encoded signature bytes.
	Base64 string
}

// An UnverifiedNoteError indicates that the note
// successfully parsed but had no verifiable signatures.
type UnverifiedNoteError struct {
	Note *Note
}

func (e *UnverifiedNoteError) Error() string {
	return "note has no verifiable signatures"
}

// An InvalidSignatureError indicates that the given key was known
// and the associated Verifier rejected the signature.
type InvalidSignatureError struct {
	Name string
	Hash uint32
}

func (e *InvalidSignatureError) Error() string {
	return fmt.Sprintf("invalid signature for key %s+%08x", e.Name, e.Hash)
}

var (
	errMalformedNote      = errors.New("malformed note")
	errInvalidSigner      = errors.New("invalid signer")
	errMismatchedVerifier = errors.New("verifier name or hash doesn't match signature")

	sigSplit  = []byte("\n\n")
	sigPrefix = []byte("— ")
)

// Open opens and parses the message msg, checking signatures from the known verifiers.
//
// For each signature in the message, Open calls known.Verifier to find a verifier.
// If known.Verifier returns a verifier and the verifier accepts the signature,
// Open records the signature in the returned note's Sigs field.
// If known.Verifier returns a verifier but the verifier rejects the signature,
// Open returns an InvalidSignatureError.
// If known.Verifier returns an UnknownVerifierError,
// Open records the signature in the returned note's UnverifiedSigs field.
// If known.Verifier returns any other error, Open returns that error.
//
// If no known verifier has signed an otherwise valid note,
// Open returns an UnverifiedNoteError.
// In this case, the unverified note can be fetched from inside the error.
func Open(msg []byte, known Verifiers) (*Note, error) {
	if known == nil {
		// Treat nil Verifiers as empty list, to produce useful error instead of crash.
		known = VerifierList()
	}

	// Must have valid UTF-8 with no non-newline ASCII control characters.
	for i := 0; i < len(msg); {
		r, size := utf8.DecodeRune(msg[i:])
		if r < 0x20 && r != '\n' || r == utf8.RuneError && size == 1 {
			return nil, errMalformedNote
		}
		i += size
	}

	// Must end with signature block preceded by blank line.
	split := bytes.LastIndex(msg, sigSplit)
	if split < 0 {
		return nil, errMalformedNote
	}
	text, sigs := msg[:split+1], msg[split+2:]
	if len(sigs) == 0 || sigs[len(sigs)-1] != '\n' {
		return nil, errMalformedNote
	}

	n := &Note{
		Text: string(text),
	}

	// Parse and verify signatures.
	// Ignore duplicate signatures.
	seen := make(map[nameHash]bool)
	seenUnverified := make(map[string]bool)
	numSig := 0
	for len(sigs) > 0 {
		// Pull out next signature line.
		// We know sigs[len(sigs)-1] == '\n', so IndexByte always finds one.
		i := bytes.IndexByte(sigs, '\n')
		line := sigs[:i]
		sigs = sigs[i+1:]

		if !bytes.HasPrefix(line, sigPrefix) {
			return nil, errMalformedNote
		}
		line = line[len(sigPrefix):]
		name, b64 := chop(string(line), " ")
		sig, err := base64.StdEncoding.DecodeString(b64)
		if err != nil || !isValidName(name) || b64 == "" || len(sig) < 5 {
			return nil, errMalformedNote
		}
		hash := binary.BigEndian.Uint32(sig[0:4])
		sig = sig[4:]

		if numSig++; numSig > 100 {
			// Avoid spending forever parsing a note with many signatures.
			return nil, errMalformedNote
		}

		v, err := known.Verifier(name, hash)
		if _, ok := err.(*UnknownVerifierError); ok {
			// Drop repeated identical unverified signatures.
			if seenUnverified[string(line)] {
				continue
			}
			seenUnverified[string(line)] = true
			n.UnverifiedSigs = append(n.UnverifiedSigs, Signature{Name: name, Hash: hash, Base64: b64})
			continue
		}
		if err != nil {
			return nil, err
		}

		// Check that known.Verifier returned the right verifier.
		if v.Name() != name || v.KeyHash() != hash {
			return nil, errMismatchedVerifier
		}

		// Drop repeated signatures by a single verifier.
		if seen[nameHash{name, hash}] {
			continue
		}
		seen[nameHash{name, hash}] = true

		ok := v.Verify(text, sig)
		if !ok {
			return nil, &InvalidSignatureError{name, hash}
		}

		n.Sigs = append(n.Sigs, Signature{Name: name, Hash: hash, Base64: b64})
	}

	// Parsed and verified all the signatures.
	if len(n.Sigs) == 0 {
		return nil, &UnverifiedNoteError{n}
	}
	return n, nil
}

// Sign signs the note with the given signers and returns the encoded message.
// The new signatures from signers are listed in the encoded message after
// the existing signatures already present in n.Sigs.
// If any signer uses the same key as an existing signature,
// the existing signature is elided from the output.
func Sign(n *Note, signers ...Signer) ([]byte, error) {
	var buf bytes.Buffer
	if !strings.HasSuffix(n.Text, "\n") {
		return nil, errMalformedNote
	}
	buf.WriteString(n.Text)

	// Prepare signatures.
	var sigs bytes.Buffer
	have := make(map[nameHash]bool)
	for _, s := range signers {
		name := s.Name()
		hash := s.KeyHash()
		have[nameHash{name, hash}] = true
		if !isValidName(name) {
			return nil, errInvalidSigner
		}

		sig, err := s.Sign(buf.Bytes()) // buf holds n.Text
		if err != nil {
			return nil, err
		}

		var hbuf [4]byte
		binary.BigEndian.PutUint32(hbuf[:], hash)
		b64 := base64.StdEncoding.EncodeToString(append(hbuf[:], sig...))
		sigs.WriteString("— ")
		sigs.WriteString(name)
		sigs.WriteString(" ")
		sigs.WriteString(b64)
		sigs.WriteString("\n")
	}

	buf.WriteString("\n")

	// Emit existing signatures not replaced by new ones.
	for _, list := range [][]Signature{n.Sigs, n.UnverifiedSigs} {
		for _, sig := range list {
			name, hash := sig.Name, sig.Hash
			if !isValidName(name) {
				return nil, errMalformedNote
			}
			if have[nameHash{name, hash}] {
				continue
			}
			// Double-check hash against base64.
			raw, err := base64.StdEncoding.DecodeString(sig.Base64)
			if err != nil || len(raw) < 4 || binary.BigEndian.Uint32(raw) != hash {
				return nil, errMalformedNote
			}
			buf.WriteString("— ")
			buf.WriteString(sig.Name)
			buf.WriteString(" ")
			buf.WriteString(sig.Base64)
			buf.WriteString("\n")
		}
	}
	buf.Write(sigs.Bytes())

	return buf.Bytes(), nil
}
