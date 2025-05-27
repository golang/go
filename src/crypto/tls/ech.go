// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"bytes"
	"crypto/internal/hpke"
	"errors"
	"fmt"
	"slices"
	"strings"

	"golang.org/x/crypto/cryptobyte"
)

// sortedSupportedAEADs is just a sorted version of hpke.SupportedAEADS.
// We need this so that when we insert them into ECHConfigs the ordering
// is stable.
var sortedSupportedAEADs []uint16

func init() {
	for aeadID := range hpke.SupportedAEADs {
		sortedSupportedAEADs = append(sortedSupportedAEADs, aeadID)
	}
	slices.Sort(sortedSupportedAEADs)
}

type echCipher struct {
	KDFID  uint16
	AEADID uint16
}

type echExtension struct {
	Type uint16
	Data []byte
}

type echConfig struct {
	raw []byte

	Version uint16
	Length  uint16

	ConfigID             uint8
	KemID                uint16
	PublicKey            []byte
	SymmetricCipherSuite []echCipher

	MaxNameLength uint8
	PublicName    []byte
	Extensions    []echExtension
}

var errMalformedECHConfigList = errors.New("tls: malformed ECHConfigList")

type echConfigErr struct {
	field string
}

func (e *echConfigErr) Error() string {
	if e.field == "" {
		return "tls: malformed ECHConfig"
	}
	return fmt.Sprintf("tls: malformed ECHConfig, invalid %s field", e.field)
}

func parseECHConfig(enc []byte) (skip bool, ec echConfig, err error) {
	s := cryptobyte.String(enc)
	ec.raw = []byte(enc)
	if !s.ReadUint16(&ec.Version) {
		return false, echConfig{}, &echConfigErr{"version"}
	}
	if !s.ReadUint16(&ec.Length) {
		return false, echConfig{}, &echConfigErr{"length"}
	}
	if len(ec.raw) < int(ec.Length)+4 {
		return false, echConfig{}, &echConfigErr{"length"}
	}
	ec.raw = ec.raw[:ec.Length+4]
	if ec.Version != extensionEncryptedClientHello {
		s.Skip(int(ec.Length))
		return true, echConfig{}, nil
	}
	if !s.ReadUint8(&ec.ConfigID) {
		return false, echConfig{}, &echConfigErr{"config_id"}
	}
	if !s.ReadUint16(&ec.KemID) {
		return false, echConfig{}, &echConfigErr{"kem_id"}
	}
	if !readUint16LengthPrefixed(&s, &ec.PublicKey) {
		return false, echConfig{}, &echConfigErr{"public_key"}
	}
	var cipherSuites cryptobyte.String
	if !s.ReadUint16LengthPrefixed(&cipherSuites) {
		return false, echConfig{}, &echConfigErr{"cipher_suites"}
	}
	for !cipherSuites.Empty() {
		var c echCipher
		if !cipherSuites.ReadUint16(&c.KDFID) {
			return false, echConfig{}, &echConfigErr{"cipher_suites kdf_id"}
		}
		if !cipherSuites.ReadUint16(&c.AEADID) {
			return false, echConfig{}, &echConfigErr{"cipher_suites aead_id"}
		}
		ec.SymmetricCipherSuite = append(ec.SymmetricCipherSuite, c)
	}
	if !s.ReadUint8(&ec.MaxNameLength) {
		return false, echConfig{}, &echConfigErr{"maximum_name_length"}
	}
	var publicName cryptobyte.String
	if !s.ReadUint8LengthPrefixed(&publicName) {
		return false, echConfig{}, &echConfigErr{"public_name"}
	}
	ec.PublicName = publicName
	var extensions cryptobyte.String
	if !s.ReadUint16LengthPrefixed(&extensions) {
		return false, echConfig{}, &echConfigErr{"extensions"}
	}
	for !extensions.Empty() {
		var e echExtension
		if !extensions.ReadUint16(&e.Type) {
			return false, echConfig{}, &echConfigErr{"extensions type"}
		}
		if !extensions.ReadUint16LengthPrefixed((*cryptobyte.String)(&e.Data)) {
			return false, echConfig{}, &echConfigErr{"extensions data"}
		}
		ec.Extensions = append(ec.Extensions, e)
	}

	return false, ec, nil
}

// parseECHConfigList parses a draft-ietf-tls-esni-18 ECHConfigList, returning a
// slice of parsed ECHConfigs, in the same order they were parsed, or an error
// if the list is malformed.
func parseECHConfigList(data []byte) ([]echConfig, error) {
	s := cryptobyte.String(data)
	var length uint16
	if !s.ReadUint16(&length) {
		return nil, errMalformedECHConfigList
	}
	if length != uint16(len(data)-2) {
		return nil, errMalformedECHConfigList
	}
	var configs []echConfig
	for len(s) > 0 {
		if len(s) < 4 {
			return nil, errors.New("tls: malformed ECHConfig")
		}
		configLen := uint16(s[2])<<8 | uint16(s[3])
		skip, ec, err := parseECHConfig(s)
		if err != nil {
			return nil, err
		}
		s = s[configLen+4:]
		if !skip {
			configs = append(configs, ec)
		}
	}
	return configs, nil
}

func pickECHConfig(list []echConfig) *echConfig {
	for _, ec := range list {
		if _, ok := hpke.SupportedKEMs[ec.KemID]; !ok {
			continue
		}
		var validSCS bool
		for _, cs := range ec.SymmetricCipherSuite {
			if _, ok := hpke.SupportedAEADs[cs.AEADID]; !ok {
				continue
			}
			if _, ok := hpke.SupportedKDFs[cs.KDFID]; !ok {
				continue
			}
			validSCS = true
			break
		}
		if !validSCS {
			continue
		}
		if !validDNSName(string(ec.PublicName)) {
			continue
		}
		var unsupportedExt bool
		for _, ext := range ec.Extensions {
			// If high order bit is set to 1 the extension is mandatory.
			// Since we don't support any extensions, if we see a mandatory
			// bit, we skip the config.
			if ext.Type&uint16(1<<15) != 0 {
				unsupportedExt = true
			}
		}
		if unsupportedExt {
			continue
		}
		return &ec
	}
	return nil
}

func pickECHCipherSuite(suites []echCipher) (echCipher, error) {
	for _, s := range suites {
		// NOTE: all of the supported AEADs and KDFs are fine, rather than
		// imposing some sort of preference here, we just pick the first valid
		// suite.
		if _, ok := hpke.SupportedAEADs[s.AEADID]; !ok {
			continue
		}
		if _, ok := hpke.SupportedKDFs[s.KDFID]; !ok {
			continue
		}
		return s, nil
	}
	return echCipher{}, errors.New("tls: no supported symmetric ciphersuites for ECH")
}

func encodeInnerClientHello(inner *clientHelloMsg, maxNameLength int) ([]byte, error) {
	h, err := inner.marshalMsg(true)
	if err != nil {
		return nil, err
	}
	h = h[4:] // strip four byte prefix

	var paddingLen int
	if inner.serverName != "" {
		paddingLen = max(0, maxNameLength-len(inner.serverName))
	} else {
		paddingLen = maxNameLength + 9
	}
	paddingLen = 31 - ((len(h) + paddingLen - 1) % 32)

	return append(h, make([]byte, paddingLen)...), nil
}

func skipUint8LengthPrefixed(s *cryptobyte.String) bool {
	var skip uint8
	if !s.ReadUint8(&skip) {
		return false
	}
	return s.Skip(int(skip))
}

func skipUint16LengthPrefixed(s *cryptobyte.String) bool {
	var skip uint16
	if !s.ReadUint16(&skip) {
		return false
	}
	return s.Skip(int(skip))
}

type rawExtension struct {
	extType uint16
	data    []byte
}

func extractRawExtensions(hello *clientHelloMsg) ([]rawExtension, error) {
	s := cryptobyte.String(hello.original)
	if !s.Skip(4+2+32) || // header, version, random
		!skipUint8LengthPrefixed(&s) || // session ID
		!skipUint16LengthPrefixed(&s) || // cipher suites
		!skipUint8LengthPrefixed(&s) { // compression methods
		return nil, errors.New("tls: malformed outer client hello")
	}
	var rawExtensions []rawExtension
	var extensions cryptobyte.String
	if !s.ReadUint16LengthPrefixed(&extensions) {
		return nil, errors.New("tls: malformed outer client hello")
	}

	for !extensions.Empty() {
		var extension uint16
		var extData cryptobyte.String
		if !extensions.ReadUint16(&extension) ||
			!extensions.ReadUint16LengthPrefixed(&extData) {
			return nil, errors.New("tls: invalid inner client hello")
		}
		rawExtensions = append(rawExtensions, rawExtension{extension, extData})
	}
	return rawExtensions, nil
}

func decodeInnerClientHello(outer *clientHelloMsg, encoded []byte) (*clientHelloMsg, error) {
	// Reconstructing the inner client hello from its encoded form is somewhat
	// complicated. It is missing its header (message type and length), session
	// ID, and the extensions may be compressed. Since we need to put the
	// extensions back in the same order as they were in the raw outer hello,
	// and since we don't store the raw extensions, or the order we parsed them
	// in, we need to reparse the raw extensions from the outer hello in order
	// to properly insert them into the inner hello. This _should_ result in raw
	// bytes which match the hello as it was generated by the client.
	innerReader := cryptobyte.String(encoded)
	var versionAndRandom, sessionID, cipherSuites, compressionMethods []byte
	var extensions cryptobyte.String
	if !innerReader.ReadBytes(&versionAndRandom, 2+32) ||
		!readUint8LengthPrefixed(&innerReader, &sessionID) ||
		len(sessionID) != 0 ||
		!readUint16LengthPrefixed(&innerReader, &cipherSuites) ||
		!readUint8LengthPrefixed(&innerReader, &compressionMethods) ||
		!innerReader.ReadUint16LengthPrefixed(&extensions) {
		return nil, errors.New("tls: invalid inner client hello")
	}

	// The specification says we must verify that the trailing padding is all
	// zeros. This is kind of weird for TLS messages, where we generally just
	// throw away any trailing garbage.
	for _, p := range innerReader {
		if p != 0 {
			return nil, errors.New("tls: invalid inner client hello")
		}
	}

	rawOuterExts, err := extractRawExtensions(outer)
	if err != nil {
		return nil, err
	}

	recon := cryptobyte.NewBuilder(nil)
	recon.AddUint8(typeClientHello)
	recon.AddUint24LengthPrefixed(func(recon *cryptobyte.Builder) {
		recon.AddBytes(versionAndRandom)
		recon.AddUint8LengthPrefixed(func(recon *cryptobyte.Builder) {
			recon.AddBytes(outer.sessionId)
		})
		recon.AddUint16LengthPrefixed(func(recon *cryptobyte.Builder) {
			recon.AddBytes(cipherSuites)
		})
		recon.AddUint8LengthPrefixed(func(recon *cryptobyte.Builder) {
			recon.AddBytes(compressionMethods)
		})
		recon.AddUint16LengthPrefixed(func(recon *cryptobyte.Builder) {
			for !extensions.Empty() {
				var extension uint16
				var extData cryptobyte.String
				if !extensions.ReadUint16(&extension) ||
					!extensions.ReadUint16LengthPrefixed(&extData) {
					recon.SetError(errors.New("tls: invalid inner client hello"))
					return
				}
				if extension == extensionECHOuterExtensions {
					if !extData.ReadUint8LengthPrefixed(&extData) {
						recon.SetError(errors.New("tls: invalid inner client hello"))
						return
					}
					var i int
					for !extData.Empty() {
						var extType uint16
						if !extData.ReadUint16(&extType) {
							recon.SetError(errors.New("tls: invalid inner client hello"))
							return
						}
						if extType == extensionEncryptedClientHello {
							recon.SetError(errors.New("tls: invalid outer extensions"))
							return
						}
						for ; i <= len(rawOuterExts); i++ {
							if i == len(rawOuterExts) {
								recon.SetError(errors.New("tls: invalid outer extensions"))
								return
							}
							if rawOuterExts[i].extType == extType {
								break
							}
						}
						recon.AddUint16(rawOuterExts[i].extType)
						recon.AddUint16LengthPrefixed(func(recon *cryptobyte.Builder) {
							recon.AddBytes(rawOuterExts[i].data)
						})
					}
				} else {
					recon.AddUint16(extension)
					recon.AddUint16LengthPrefixed(func(recon *cryptobyte.Builder) {
						recon.AddBytes(extData)
					})
				}
			}
		})
	})

	reconBytes, err := recon.Bytes()
	if err != nil {
		return nil, err
	}
	inner := &clientHelloMsg{}
	if !inner.unmarshal(reconBytes) {
		return nil, errors.New("tls: invalid reconstructed inner client hello")
	}

	if !bytes.Equal(inner.encryptedClientHello, []byte{uint8(innerECHExt)}) {
		return nil, errInvalidECHExt
	}

	hasTLS13 := false
	for _, v := range inner.supportedVersions {
		// Skip GREASE values (values of the form 0x?A0A).
		// GREASE (Generate Random Extensions And Sustain Extensibility) is a mechanism used by
		// browsers like Chrome to ensure TLS implementations correctly ignore unknown values.
		// GREASE values follow a specific pattern: 0x?A0A, where ? can be any hex digit.
		// These values should be ignored when processing supported TLS versions.
		if v&0x0F0F == 0x0A0A && v&0xff == v>>8 {
			continue
		}

		// Ensure at least TLS 1.3 is offered.
		if v == VersionTLS13 {
			hasTLS13 = true
		} else if v < VersionTLS13 {
			// Reject if any non-GREASE value is below TLS 1.3, as ECH requires TLS 1.3+.
			return nil, errors.New("tls: client sent encrypted_client_hello extension with unsupported versions")
		}
	}

	if !hasTLS13 {
		return nil, errors.New("tls: client sent encrypted_client_hello extension but did not offer TLS 1.3")
	}

	return inner, nil
}

func decryptECHPayload(context *hpke.Recipient, hello, payload []byte) ([]byte, error) {
	outerAAD := bytes.Replace(hello[4:], payload, make([]byte, len(payload)), 1)
	return context.Open(outerAAD, payload)
}

func generateOuterECHExt(id uint8, kdfID, aeadID uint16, encodedKey []byte, payload []byte) ([]byte, error) {
	var b cryptobyte.Builder
	b.AddUint8(0) // outer
	b.AddUint16(kdfID)
	b.AddUint16(aeadID)
	b.AddUint8(id)
	b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) { b.AddBytes(encodedKey) })
	b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) { b.AddBytes(payload) })
	return b.Bytes()
}

func computeAndUpdateOuterECHExtension(outer, inner *clientHelloMsg, ech *echClientContext, useKey bool) error {
	var encapKey []byte
	if useKey {
		encapKey = ech.encapsulatedKey
	}
	encodedInner, err := encodeInnerClientHello(inner, int(ech.config.MaxNameLength))
	if err != nil {
		return err
	}
	// NOTE: the tag lengths for all of the supported AEADs are the same (16
	// bytes), so we have hardcoded it here. If we add support for another AEAD
	// with a different tag length, we will need to change this.
	encryptedLen := len(encodedInner) + 16 // AEAD tag length
	outer.encryptedClientHello, err = generateOuterECHExt(ech.config.ConfigID, ech.kdfID, ech.aeadID, encapKey, make([]byte, encryptedLen))
	if err != nil {
		return err
	}
	serializedOuter, err := outer.marshal()
	if err != nil {
		return err
	}
	serializedOuter = serializedOuter[4:] // strip the four byte prefix
	encryptedInner, err := ech.hpkeContext.Seal(serializedOuter, encodedInner)
	if err != nil {
		return err
	}
	outer.encryptedClientHello, err = generateOuterECHExt(ech.config.ConfigID, ech.kdfID, ech.aeadID, encapKey, encryptedInner)
	if err != nil {
		return err
	}
	return nil
}

// validDNSName is a rather rudimentary check for the validity of a DNS name.
// This is used to check if the public_name in a ECHConfig is valid when we are
// picking a config. This can be somewhat lax because even if we pick a
// valid-looking name, the DNS layer will later reject it anyway.
func validDNSName(name string) bool {
	if len(name) > 253 {
		return false
	}
	labels := strings.Split(name, ".")
	if len(labels) <= 1 {
		return false
	}
	for _, l := range labels {
		labelLen := len(l)
		if labelLen == 0 {
			return false
		}
		for i, r := range l {
			if r == '-' && (i == 0 || i == labelLen-1) {
				return false
			}
			if (r < '0' || r > '9') && (r < 'a' || r > 'z') && (r < 'A' || r > 'Z') && r != '-' {
				return false
			}
		}
	}
	return true
}

// ECHRejectionError is the error type returned when ECH is rejected by a remote
// server. If the server offered a ECHConfigList to use for retries, the
// RetryConfigList field will contain this list.
//
// The client may treat an ECHRejectionError with an empty set of RetryConfigs
// as a secure signal from the server.
type ECHRejectionError struct {
	RetryConfigList []byte
}

func (e *ECHRejectionError) Error() string {
	return "tls: server rejected ECH"
}

var errMalformedECHExt = errors.New("tls: malformed encrypted_client_hello extension")
var errInvalidECHExt = errors.New("tls: client sent invalid encrypted_client_hello extension")

type echExtType uint8

const (
	innerECHExt echExtType = 1
	outerECHExt echExtType = 0
)

func parseECHExt(ext []byte) (echType echExtType, cs echCipher, configID uint8, encap []byte, payload []byte, err error) {
	data := make([]byte, len(ext))
	copy(data, ext)
	s := cryptobyte.String(data)
	var echInt uint8
	if !s.ReadUint8(&echInt) {
		err = errMalformedECHExt
		return
	}
	echType = echExtType(echInt)
	if echType == innerECHExt {
		if !s.Empty() {
			err = errMalformedECHExt
			return
		}
		return echType, cs, 0, nil, nil, nil
	}
	if echType != outerECHExt {
		err = errInvalidECHExt
		return
	}
	if !s.ReadUint16(&cs.KDFID) {
		err = errMalformedECHExt
		return
	}
	if !s.ReadUint16(&cs.AEADID) {
		err = errMalformedECHExt
		return
	}
	if !s.ReadUint8(&configID) {
		err = errMalformedECHExt
		return
	}
	if !readUint16LengthPrefixed(&s, &encap) {
		err = errMalformedECHExt
		return
	}
	if !readUint16LengthPrefixed(&s, &payload) {
		err = errMalformedECHExt
		return
	}

	// NOTE: clone encap and payload so that mutating them does not mutate the
	// raw extension bytes.
	return echType, cs, configID, bytes.Clone(encap), bytes.Clone(payload), nil
}

func marshalEncryptedClientHelloConfigList(configs []EncryptedClientHelloKey) ([]byte, error) {
	builder := cryptobyte.NewBuilder(nil)
	builder.AddUint16LengthPrefixed(func(builder *cryptobyte.Builder) {
		for _, c := range configs {
			builder.AddBytes(c.Config)
		}
	})
	return builder.Bytes()
}

func (c *Conn) processECHClientHello(outer *clientHelloMsg, echKeys []EncryptedClientHelloKey) (*clientHelloMsg, *echServerContext, error) {
	echType, echCiphersuite, configID, encap, payload, err := parseECHExt(outer.encryptedClientHello)
	if err != nil {
		if errors.Is(err, errInvalidECHExt) {
			c.sendAlert(alertIllegalParameter)
		} else {
			c.sendAlert(alertDecodeError)
		}

		return nil, nil, errInvalidECHExt
	}

	if echType == innerECHExt {
		return outer, &echServerContext{inner: true}, nil
	}

	if len(echKeys) == 0 {
		return outer, nil, nil
	}

	for _, echKey := range echKeys {
		skip, config, err := parseECHConfig(echKey.Config)
		if err != nil || skip {
			c.sendAlert(alertInternalError)
			return nil, nil, fmt.Errorf("tls: invalid EncryptedClientHelloKeys Config: %s", err)
		}
		if skip {
			continue
		}
		echPriv, err := hpke.ParseHPKEPrivateKey(config.KemID, echKey.PrivateKey)
		if err != nil {
			c.sendAlert(alertInternalError)
			return nil, nil, fmt.Errorf("tls: invalid EncryptedClientHelloKeys PrivateKey: %s", err)
		}
		info := append([]byte("tls ech\x00"), echKey.Config...)
		hpkeContext, err := hpke.SetupRecipient(hpke.DHKEM_X25519_HKDF_SHA256, echCiphersuite.KDFID, echCiphersuite.AEADID, echPriv, info, encap)
		if err != nil {
			// attempt next trial decryption
			continue
		}

		encodedInner, err := decryptECHPayload(hpkeContext, outer.original, payload)
		if err != nil {
			// attempt next trial decryption
			continue
		}

		// NOTE: we do not enforce that the sent server_name matches the ECH
		// configs PublicName, since this is not particularly important, and
		// the client already had to know what it was in order to properly
		// encrypt the payload. This is only a MAY in the spec, so we're not
		// doing anything revolutionary.

		echInner, err := decodeInnerClientHello(outer, encodedInner)
		if err != nil {
			c.sendAlert(alertIllegalParameter)
			return nil, nil, errInvalidECHExt
		}

		c.echAccepted = true

		return echInner, &echServerContext{
			hpkeContext: hpkeContext,
			configID:    configID,
			ciphersuite: echCiphersuite,
		}, nil
	}

	return outer, nil, nil
}

func buildRetryConfigList(keys []EncryptedClientHelloKey) ([]byte, error) {
	var atLeastOneRetryConfig bool
	var retryBuilder cryptobyte.Builder
	retryBuilder.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
		for _, c := range keys {
			if !c.SendAsRetry {
				continue
			}
			atLeastOneRetryConfig = true
			b.AddBytes(c.Config)
		}
	})
	if !atLeastOneRetryConfig {
		return nil, nil
	}
	return retryBuilder.Bytes()
}
