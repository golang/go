// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fipstest

// A module wrapper adapting the Go FIPS module to the protocol used by the
// BoringSSL project's `acvptool`.
//
// The `acvptool` "lowers" the NIST ACVP server JSON test vectors into a simpler
// stdin/stdout protocol that can be implemented by a module shim. The tool
// will fork this binary, request the supported configuration, and then provide
// test cases over stdin, expecting results to be returned on stdout.
//
// See "Testing other FIPS modules"[0] from the BoringSSL ACVP.md documentation
// for a more detailed description of the protocol used between the acvptool
// and module wrappers.
//
// [0]: https://boringssl.googlesource.com/boringssl/+/refs/heads/master/util/fipstools/acvp/ACVP.md#testing-other-fips-modules

import (
	"bufio"
	"bytes"
	"crypto/elliptic"
	"crypto/internal/cryptotest"
	"crypto/internal/fips140"
	"crypto/internal/fips140/aes"
	"crypto/internal/fips140/aes/gcm"
	"crypto/internal/fips140/ecdsa"
	"crypto/internal/fips140/ed25519"
	"crypto/internal/fips140/edwards25519"
	"crypto/internal/fips140/hmac"
	"crypto/internal/fips140/mlkem"
	"crypto/internal/fips140/pbkdf2"
	"crypto/internal/fips140/sha256"
	"crypto/internal/fips140/sha3"
	"crypto/internal/fips140/sha512"
	"crypto/internal/fips140/subtle"
	"crypto/rand"
	_ "embed"
	"encoding/binary"
	"errors"
	"fmt"
	"internal/testenv"
	"io"
	"math/big"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestMain(m *testing.M) {
	if os.Getenv("ACVP_WRAPPER") == "1" {
		wrapperMain()
	} else {
		os.Exit(m.Run())
	}
}

func wrapperMain() {
	if err := processingLoop(bufio.NewReader(os.Stdin), os.Stdout); err != nil {
		fmt.Fprintf(os.Stderr, "processing error: %v\n", err)
		os.Exit(1)
	}
}

type request struct {
	name string
	args [][]byte
}

type commandHandler func([][]byte) ([][]byte, error)

type command struct {
	// requiredArgs enforces that an exact number of arguments are provided to the handler.
	requiredArgs int
	handler      commandHandler
}

type ecdsaSigType int

const (
	ecdsaSigTypeNormal ecdsaSigType = iota
	ecdsaSigTypeDeterministic
)

type aesDirection int

const (
	aesEncrypt aesDirection = iota
	aesDecrypt
)

var (
	// SHA2 algorithm capabilities:
	//   https://pages.nist.gov/ACVP/draft-celi-acvp-sha.html#section-7.2
	// HMAC algorithm capabilities:
	//   https://pages.nist.gov/ACVP/draft-fussell-acvp-mac.html#section-7
	// PBKDF2 algorithm capabilities:
	//   https://pages.nist.gov/ACVP/draft-celi-acvp-pbkdf.html#section-7.3
	// ML-KEM algorithm capabilities:
	//   https://pages.nist.gov/ACVP/draft-celi-acvp-ml-kem.html#section-7.3
	// HMAC DRBG algorithm capabilities:
	//   https://pages.nist.gov/ACVP/draft-vassilev-acvp-drbg.html#section-7.2
	// EDDSA algorithm capabilities:
	//   https://pages.nist.gov/ACVP/draft-celi-acvp-eddsa.html#section-7
	// ECDSA and DetECDSA algorithm capabilities:
	//   https://pages.nist.gov/ACVP/draft-fussell-acvp-ecdsa.html#section-7
	// AES algorithm capabilities:
	//   https://pages.nist.gov/ACVP/draft-celi-acvp-symmetric.html#section-7.3
	//go:embed acvp_capabilities.json
	capabilitiesJson []byte

	// commands should reflect what config says we support. E.g. adding a command here will be a NOP
	// unless the configuration/acvp_capabilities.json indicates the command's associated algorithm
	// is supported.
	commands = map[string]command{
		"getConfig": cmdGetConfig(),

		"SHA2-224":         cmdHashAft(sha256.New224()),
		"SHA2-224/MCT":     cmdHashMct(sha256.New224()),
		"SHA2-256":         cmdHashAft(sha256.New()),
		"SHA2-256/MCT":     cmdHashMct(sha256.New()),
		"SHA2-384":         cmdHashAft(sha512.New384()),
		"SHA2-384/MCT":     cmdHashMct(sha512.New384()),
		"SHA2-512":         cmdHashAft(sha512.New()),
		"SHA2-512/MCT":     cmdHashMct(sha512.New()),
		"SHA2-512/224":     cmdHashAft(sha512.New512_224()),
		"SHA2-512/224/MCT": cmdHashMct(sha512.New512_224()),
		"SHA2-512/256":     cmdHashAft(sha512.New512_256()),
		"SHA2-512/256/MCT": cmdHashMct(sha512.New512_256()),

		"SHA3-256":     cmdHashAft(sha3.New256()),
		"SHA3-256/MCT": cmdSha3Mct(sha3.New256()),
		"SHA3-224":     cmdHashAft(sha3.New224()),
		"SHA3-224/MCT": cmdSha3Mct(sha3.New224()),
		"SHA3-384":     cmdHashAft(sha3.New384()),
		"SHA3-384/MCT": cmdSha3Mct(sha3.New384()),
		"SHA3-512":     cmdHashAft(sha3.New512()),
		"SHA3-512/MCT": cmdSha3Mct(sha3.New512()),

		"HMAC-SHA2-224":     cmdHmacAft(func() fips140.Hash { return sha256.New224() }),
		"HMAC-SHA2-256":     cmdHmacAft(func() fips140.Hash { return sha256.New() }),
		"HMAC-SHA2-384":     cmdHmacAft(func() fips140.Hash { return sha512.New384() }),
		"HMAC-SHA2-512":     cmdHmacAft(func() fips140.Hash { return sha512.New() }),
		"HMAC-SHA2-512/224": cmdHmacAft(func() fips140.Hash { return sha512.New512_224() }),
		"HMAC-SHA2-512/256": cmdHmacAft(func() fips140.Hash { return sha512.New512_256() }),
		"HMAC-SHA3-224":     cmdHmacAft(func() fips140.Hash { return sha3.New224() }),
		"HMAC-SHA3-256":     cmdHmacAft(func() fips140.Hash { return sha3.New256() }),
		"HMAC-SHA3-384":     cmdHmacAft(func() fips140.Hash { return sha3.New384() }),
		"HMAC-SHA3-512":     cmdHmacAft(func() fips140.Hash { return sha3.New512() }),

		"PBKDF": cmdPbkdf(),

		"ML-KEM-768/keyGen":  cmdMlKem768KeyGenAft(),
		"ML-KEM-768/encap":   cmdMlKem768EncapAft(),
		"ML-KEM-768/decap":   cmdMlKem768DecapAft(),
		"ML-KEM-1024/keyGen": cmdMlKem1024KeyGenAft(),
		"ML-KEM-1024/encap":  cmdMlKem1024EncapAft(),
		"ML-KEM-1024/decap":  cmdMlKem1024DecapAft(),

		"hmacDRBG/SHA2-224":     cmdHmacDrbgAft(func() fips140.Hash { return sha256.New224() }),
		"hmacDRBG/SHA2-256":     cmdHmacDrbgAft(func() fips140.Hash { return sha256.New() }),
		"hmacDRBG/SHA2-384":     cmdHmacDrbgAft(func() fips140.Hash { return sha512.New384() }),
		"hmacDRBG/SHA2-512":     cmdHmacDrbgAft(func() fips140.Hash { return sha512.New() }),
		"hmacDRBG/SHA2-512/224": cmdHmacDrbgAft(func() fips140.Hash { return sha512.New512_224() }),
		"hmacDRBG/SHA2-512/256": cmdHmacDrbgAft(func() fips140.Hash { return sha512.New512_256() }),
		"hmacDRBG/SHA3-224":     cmdHmacDrbgAft(func() fips140.Hash { return sha3.New224() }),
		"hmacDRBG/SHA3-256":     cmdHmacDrbgAft(func() fips140.Hash { return sha3.New256() }),
		"hmacDRBG/SHA3-384":     cmdHmacDrbgAft(func() fips140.Hash { return sha3.New384() }),
		"hmacDRBG/SHA3-512":     cmdHmacDrbgAft(func() fips140.Hash { return sha3.New512() }),

		"EDDSA/keyGen": cmdEddsaKeyGenAft(),
		"EDDSA/keyVer": cmdEddsaKeyVerAft(),
		"EDDSA/sigGen": cmdEddsaSigGenAftBft(),
		"EDDSA/sigVer": cmdEddsaSigVerAft(),

		"ECDSA/keyGen":    cmdEcdsaKeyGenAft(),
		"ECDSA/keyVer":    cmdEcdsaKeyVerAft(),
		"ECDSA/sigGen":    cmdEcdsaSigGenAft(ecdsaSigTypeNormal),
		"ECDSA/sigVer":    cmdEcdsaSigVerAft(),
		"DetECDSA/sigGen": cmdEcdsaSigGenAft(ecdsaSigTypeDeterministic),

		"AES-CBC/encrypt":        cmdAesCbc(aesEncrypt),
		"AES-CBC/decrypt":        cmdAesCbc(aesDecrypt),
		"AES-CTR/encrypt":        cmdAesCtr(aesEncrypt),
		"AES-CTR/decrypt":        cmdAesCtr(aesDecrypt),
		"AES-GCM/seal":           cmdAesGcmSeal(false),
		"AES-GCM/open":           cmdAesGcmOpen(false),
		"AES-GCM-randnonce/seal": cmdAesGcmSeal(true),
		"AES-GCM-randnonce/open": cmdAesGcmOpen(true),

		"CMAC-AES":        cmdCmacAesAft(),
		"CMAC-AES/verify": cmdCmacAesVerifyAft(),
	}
)

func processingLoop(reader io.Reader, writer io.Writer) error {
	// Per ACVP.md:
	//   The protocol is request–response: the subprocess only speaks in response to a request
	//   and there is exactly one response for every request.
	for {
		req, err := readRequest(reader)
		if errors.Is(err, io.EOF) {
			break
		} else if err != nil {
			return fmt.Errorf("reading request: %w", err)
		}

		cmd, exists := commands[req.name]
		if !exists {
			return fmt.Errorf("unknown command: %q", req.name)
		}

		if gotArgs := len(req.args); gotArgs != cmd.requiredArgs {
			return fmt.Errorf("command %q expected %d args, got %d", req.name, cmd.requiredArgs, gotArgs)
		}

		response, err := cmd.handler(req.args)
		if err != nil {
			return fmt.Errorf("command %q failed: %w", req.name, err)
		}

		if err = writeResponse(writer, response); err != nil {
			return fmt.Errorf("command %q response failed: %w", req.name, err)
		}
	}

	return nil
}

func readRequest(reader io.Reader) (*request, error) {
	// Per ACVP.md:
	//   Requests consist of one or more byte strings and responses consist
	//   of zero or more byte strings. A request contains: the number of byte
	//   strings, the length of each byte string, and the contents of each byte
	//   string. All numbers are 32-bit little-endian and values are
	//   concatenated in the order specified.
	var numArgs uint32
	if err := binary.Read(reader, binary.LittleEndian, &numArgs); err != nil {
		return nil, err
	}
	if numArgs == 0 {
		return nil, errors.New("invalid request: zero args")
	}

	args, err := readArgs(reader, numArgs)
	if err != nil {
		return nil, err
	}

	return &request{
		name: string(args[0]),
		args: args[1:],
	}, nil
}

func readArgs(reader io.Reader, requiredArgs uint32) ([][]byte, error) {
	argLengths := make([]uint32, requiredArgs)
	args := make([][]byte, requiredArgs)

	for i := range argLengths {
		if err := binary.Read(reader, binary.LittleEndian, &argLengths[i]); err != nil {
			return nil, fmt.Errorf("invalid request: failed to read %d-th arg len: %w", i, err)
		}
	}

	for i, length := range argLengths {
		buf := make([]byte, length)
		if _, err := io.ReadFull(reader, buf); err != nil {
			return nil, fmt.Errorf("invalid request: failed to read %d-th arg data: %w", i, err)
		}
		args[i] = buf
	}

	return args, nil
}

func writeResponse(writer io.Writer, args [][]byte) error {
	// See `readRequest` for details on the base format. Per ACVP.md:
	//   A response has the same format except that there may be zero byte strings
	//   and the first byte string has no special meaning.
	numArgs := uint32(len(args))
	if err := binary.Write(writer, binary.LittleEndian, numArgs); err != nil {
		return fmt.Errorf("writing arg count: %w", err)
	}

	for i, arg := range args {
		if err := binary.Write(writer, binary.LittleEndian, uint32(len(arg))); err != nil {
			return fmt.Errorf("writing %d-th arg length: %w", i, err)
		}
	}

	for i, b := range args {
		if _, err := writer.Write(b); err != nil {
			return fmt.Errorf("writing %d-th arg data: %w", i, err)
		}
	}

	return nil
}

// "All implementations must support the getConfig command
// which takes no arguments and returns a single byte string
// which is a JSON blob of ACVP algorithm configuration."
func cmdGetConfig() command {
	return command{
		handler: func(args [][]byte) ([][]byte, error) {
			return [][]byte{capabilitiesJson}, nil
		},
	}
}

// cmdHashAft returns a command handler for the specified hash
// algorithm for algorithm functional test (AFT) test cases.
//
// This shape of command expects a message as the sole argument,
// and writes the resulting digest as a response.
//
// See https://pages.nist.gov/ACVP/draft-celi-acvp-sha.html
func cmdHashAft(h fips140.Hash) command {
	return command{
		requiredArgs: 1, // Message to hash.
		handler: func(args [][]byte) ([][]byte, error) {
			h.Reset()
			h.Write(args[0])
			digest := make([]byte, 0, h.Size())
			digest = h.Sum(digest)

			return [][]byte{digest}, nil
		},
	}
}

// cmdHashMct returns a command handler for the specified hash
// algorithm for monte carlo test (MCT) test cases.
//
// This shape of command expects a seed as the sole argument,
// and writes the resulting digest as a response. It implements
// the "standard" flavour of the MCT, not the "alternative".
//
// This algorithm was ported from `HashMCT` in BSSL's `modulewrapper.cc`
// Note that it differs slightly from the upstream NIST MCT[0] algorithm
// in that it does not perform the outer 100 iterations itself. See
// footnote #1 in the ACVP.md docs[1], the acvptool handles this.
//
// [0]: https://pages.nist.gov/ACVP/draft-celi-acvp-sha.html#section-6.2
// [1]: https://boringssl.googlesource.com/boringssl/+/refs/heads/master/util/fipstools/acvp/ACVP.md#testing-other-fips-modules
func cmdHashMct(h fips140.Hash) command {
	return command{
		requiredArgs: 1, // Seed message.
		handler: func(args [][]byte) ([][]byte, error) {
			hSize := h.Size()
			seed := args[0]

			if seedLen := len(seed); seedLen != hSize {
				return nil, fmt.Errorf("invalid seed size: expected %d got %d", hSize, seedLen)
			}

			digest := make([]byte, 0, hSize)
			buf := make([]byte, 0, 3*hSize)
			buf = append(buf, seed...)
			buf = append(buf, seed...)
			buf = append(buf, seed...)

			for i := 0; i < 1000; i++ {
				h.Reset()
				h.Write(buf)
				digest = h.Sum(digest[:0])

				copy(buf, buf[hSize:])
				copy(buf[2*hSize:], digest)
			}

			return [][]byte{buf[hSize*2:]}, nil
		},
	}
}

// cmdSha3Mct returns a command handler for the specified hash
// algorithm for SHA-3 monte carlo test (MCT) test cases.
//
// This shape of command expects a seed as the sole argument,
// and writes the resulting digest as a response. It implements
// the "standard" flavour of the MCT, not the "alternative".
//
// This algorithm was ported from the "standard" MCT algorithm
// specified in  draft-celi-acvp-sha3[0]. Note this differs from
// the SHA2-* family of MCT tests handled by cmdHashMct. However,
// like that handler it does not perform the outer 100 iterations.
//
// [0]: https://pages.nist.gov/ACVP/draft-celi-acvp-sha3.html#section-6.2.1
func cmdSha3Mct(h fips140.Hash) command {
	return command{
		requiredArgs: 1, // Seed message.
		handler: func(args [][]byte) ([][]byte, error) {
			seed := args[0]
			md := make([][]byte, 1001)
			md[0] = seed

			for i := 1; i <= 1000; i++ {
				h.Reset()
				h.Write(md[i-1])
				md[i] = h.Sum(nil)
			}

			return [][]byte{md[1000]}, nil
		},
	}
}

func cmdHmacAft(h func() fips140.Hash) command {
	return command{
		requiredArgs: 2, // Message and key
		handler: func(args [][]byte) ([][]byte, error) {
			msg := args[0]
			key := args[1]
			mac := hmac.New(h, key)
			mac.Write(msg)
			return [][]byte{mac.Sum(nil)}, nil
		},
	}
}

func cmdPbkdf() command {
	return command{
		// Hash name, key length, salt, password, iteration count
		requiredArgs: 5,
		handler: func(args [][]byte) ([][]byte, error) {
			h, err := lookupHash(string(args[0]))
			if err != nil {
				return nil, fmt.Errorf("PBKDF2 failed: %w", err)
			}

			keyLen := binary.LittleEndian.Uint32(args[1]) / 8
			salt := args[2]
			password := args[3]
			iterationCount := binary.LittleEndian.Uint32(args[4])

			derivedKey, err := pbkdf2.Key(h, string(password), salt, int(iterationCount), int(keyLen))
			if err != nil {
				return nil, fmt.Errorf("PBKDF2 failed: %w", err)
			}

			return [][]byte{derivedKey}, nil
		},
	}
}

func cmdEddsaKeyGenAft() command {
	return command{
		requiredArgs: 1, // Curve name
		handler: func(args [][]byte) ([][]byte, error) {
			if string(args[0]) != "ED-25519" {
				return nil, fmt.Errorf("unsupported EDDSA curve: %q", args[0])
			}

			sk, err := ed25519.GenerateKey()
			if err != nil {
				return nil, fmt.Errorf("generating EDDSA keypair: %w", err)
			}

			// EDDSA/keyGen/AFT responses are d & q, described[0] as:
			//   d	The encoded private key point
			//   q	The encoded public key point
			//
			// Contrary to the description of a "point", d is the private key
			// seed bytes per FIPS.186-5[1] A.2.3.
			//
			// [0]: https://pages.nist.gov/ACVP/draft-celi-acvp-eddsa.html#section-9.1
			// [1]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.186-5.pdf
			return [][]byte{sk.Seed(), sk.PublicKey()}, nil
		},
	}
}

func cmdEddsaKeyVerAft() command {
	return command{
		requiredArgs: 2, // Curve name, Q
		handler: func(args [][]byte) ([][]byte, error) {
			if string(args[0]) != "ED-25519" {
				return nil, fmt.Errorf("unsupported EDDSA curve: %q", args[0])
			}

			// Verify the point is on the curve. The higher-level ed25519 API does
			// this at signature verification time so we have to use the lower-level
			// edwards25519 package to do it here in absence of a signature to verify.
			if _, err := new(edwards25519.Point).SetBytes(args[1]); err != nil {
				return [][]byte{{0}}, nil
			}

			return [][]byte{{1}}, nil
		},
	}
}

func cmdEddsaSigGenAftBft() command {
	return command{
		requiredArgs: 5, // Curve name, private key seed, message, prehash, context
		handler: func(args [][]byte) ([][]byte, error) {
			if string(args[0]) != "ED-25519" {
				return nil, fmt.Errorf("unsupported EDDSA curve: %q", args[0])
			}

			sk, err := ed25519.NewPrivateKeyFromSeed(args[1])
			if err != nil {
				return nil, fmt.Errorf("error creating private key: %w", err)
			}
			msg := args[2]
			prehash := args[3]
			context := string(args[4])

			var sig []byte
			if prehash[0] == 1 {
				h := sha512.New()
				h.Write(msg)
				msg = h.Sum(nil)

				// With ed25519 the context is only specified for sigGen tests when using prehashing.
				// See https://pages.nist.gov/ACVP/draft-celi-acvp-eddsa.html#section-8.6
				sig, err = ed25519.SignPH(sk, msg, context)
				if err != nil {
					return nil, fmt.Errorf("error signing message: %w", err)
				}
			} else {
				sig = ed25519.Sign(sk, msg)
			}

			return [][]byte{sig}, nil
		},
	}
}

func cmdEddsaSigVerAft() command {
	return command{
		requiredArgs: 5, // Curve name, message, public key, signature, prehash
		handler: func(args [][]byte) ([][]byte, error) {
			if string(args[0]) != "ED-25519" {
				return nil, fmt.Errorf("unsupported EDDSA curve: %q", args[0])
			}

			msg := args[1]
			pk, err := ed25519.NewPublicKey(args[2])
			if err != nil {
				return nil, fmt.Errorf("invalid public key: %w", err)
			}
			sig := args[3]
			prehash := args[4]

			if prehash[0] == 1 {
				h := sha512.New()
				h.Write(msg)
				msg = h.Sum(nil)
				// Context is only specified for sigGen, not sigVer.
				// See https://pages.nist.gov/ACVP/draft-celi-acvp-eddsa.html#section-8.6
				err = ed25519.VerifyPH(pk, msg, sig, "")
			} else {
				err = ed25519.Verify(pk, msg, sig)
			}

			if err != nil {
				return [][]byte{{0}}, nil
			}

			return [][]byte{{1}}, nil
		},
	}
}

func cmdEcdsaKeyGenAft() command {
	return command{
		requiredArgs: 1, // Curve name
		handler: func(args [][]byte) ([][]byte, error) {
			curve, err := lookupCurve(string(args[0]))
			if err != nil {
				return nil, err
			}

			var sk *ecdsa.PrivateKey
			switch curve.Params() {
			case elliptic.P224().Params():
				sk, err = ecdsa.GenerateKey(ecdsa.P224(), rand.Reader)
			case elliptic.P256().Params():
				sk, err = ecdsa.GenerateKey(ecdsa.P256(), rand.Reader)
			case elliptic.P384().Params():
				sk, err = ecdsa.GenerateKey(ecdsa.P384(), rand.Reader)
			case elliptic.P521().Params():
				sk, err = ecdsa.GenerateKey(ecdsa.P521(), rand.Reader)
			default:
				return nil, fmt.Errorf("unsupported curve: %v", curve)
			}

			if err != nil {
				return nil, err
			}

			pubBytes := sk.PublicKey().Bytes()
			byteLen := (curve.Params().BitSize + 7) / 8

			return [][]byte{
				sk.Bytes(),
				pubBytes[1 : 1+byteLen],
				pubBytes[1+byteLen:],
			}, nil
		},
	}
}

func cmdEcdsaKeyVerAft() command {
	return command{
		requiredArgs: 3, // Curve name, X, Y
		handler: func(args [][]byte) ([][]byte, error) {
			curve, err := lookupCurve(string(args[0]))
			if err != nil {
				return nil, err
			}

			x := new(big.Int).SetBytes(args[1])
			y := new(big.Int).SetBytes(args[2])

			if curve.IsOnCurve(x, y) {
				return [][]byte{{1}}, nil
			}

			return [][]byte{{0}}, nil
		},
	}
}

// pointFromAffine is used to convert the PublicKey to a nistec SetBytes input.
// Duplicated from crypto/ecdsa.go's pointFromAffine.
func pointFromAffine(curve elliptic.Curve, x, y *big.Int) ([]byte, error) {
	bitSize := curve.Params().BitSize
	// Reject values that would not get correctly encoded.
	if x.Sign() < 0 || y.Sign() < 0 {
		return nil, errors.New("negative coordinate")
	}
	if x.BitLen() > bitSize || y.BitLen() > bitSize {
		return nil, errors.New("overflowing coordinate")
	}
	// Encode the coordinates and let SetBytes reject invalid points.
	byteLen := (bitSize + 7) / 8
	buf := make([]byte, 1+2*byteLen)
	buf[0] = 4 // uncompressed point
	x.FillBytes(buf[1 : 1+byteLen])
	y.FillBytes(buf[1+byteLen : 1+2*byteLen])
	return buf, nil
}

func signEcdsa[P ecdsa.Point[P], H fips140.Hash](c *ecdsa.Curve[P], h func() H, sigType ecdsaSigType, q []byte, sk []byte, digest []byte) (*ecdsa.Signature, error) {
	priv, err := ecdsa.NewPrivateKey(c, sk, q)
	if err != nil {
		return nil, fmt.Errorf("invalid private key: %w", err)
	}

	var sig *ecdsa.Signature
	switch sigType {
	case ecdsaSigTypeNormal:
		sig, err = ecdsa.Sign(c, h, priv, rand.Reader, digest)
	case ecdsaSigTypeDeterministic:
		sig, err = ecdsa.SignDeterministic(c, h, priv, digest)
	default:
		return nil, fmt.Errorf("unsupported signature type: %v", sigType)
	}
	if err != nil {
		return nil, fmt.Errorf("signing failed: %w", err)
	}

	return sig, nil
}

func cmdEcdsaSigGenAft(sigType ecdsaSigType) command {
	return command{
		requiredArgs: 4, // Curve name, private key, hash name, message
		handler: func(args [][]byte) ([][]byte, error) {
			curve, err := lookupCurve(string(args[0]))
			if err != nil {
				return nil, err
			}

			sk := args[1]

			newH, err := lookupHash(string(args[2]))
			if err != nil {
				return nil, err
			}

			msg := args[3]
			hashFunc := newH()
			hashFunc.Write(msg)
			digest := hashFunc.Sum(nil)

			d := new(big.Int).SetBytes(sk)
			x, y := curve.ScalarBaseMult(d.Bytes())
			q, err := pointFromAffine(curve, x, y)
			if err != nil {
				return nil, err
			}

			var sig *ecdsa.Signature
			switch curve.Params() {
			case elliptic.P224().Params():
				sig, err = signEcdsa(ecdsa.P224(), newH, sigType, q, sk, digest)
			case elliptic.P256().Params():
				sig, err = signEcdsa(ecdsa.P256(), newH, sigType, q, sk, digest)
			case elliptic.P384().Params():
				sig, err = signEcdsa(ecdsa.P384(), newH, sigType, q, sk, digest)
			case elliptic.P521().Params():
				sig, err = signEcdsa(ecdsa.P521(), newH, sigType, q, sk, digest)
			default:
				return nil, fmt.Errorf("unsupported curve: %v", curve)
			}
			if err != nil {
				return nil, err
			}

			return [][]byte{sig.R, sig.S}, nil
		},
	}
}

func cmdEcdsaSigVerAft() command {
	return command{
		requiredArgs: 7, // Curve name, hash name, message, X, Y, R, S
		handler: func(args [][]byte) ([][]byte, error) {
			curve, err := lookupCurve(string(args[0]))
			if err != nil {
				return nil, err
			}

			newH, err := lookupHash(string(args[1]))
			if err != nil {
				return nil, err
			}

			msg := args[2]
			hashFunc := newH()
			hashFunc.Write(msg)
			digest := hashFunc.Sum(nil)

			x, y := args[3], args[4]
			q, err := pointFromAffine(curve, new(big.Int).SetBytes(x), new(big.Int).SetBytes(y))
			if err != nil {
				return nil, fmt.Errorf("invalid x/y coordinates: %v", err)
			}

			signature := &ecdsa.Signature{R: args[5], S: args[6]}

			switch curve.Params() {
			case elliptic.P224().Params():
				err = verifyEcdsa(ecdsa.P224(), q, digest, signature)
			case elliptic.P256().Params():
				err = verifyEcdsa(ecdsa.P256(), q, digest, signature)
			case elliptic.P384().Params():
				err = verifyEcdsa(ecdsa.P384(), q, digest, signature)
			case elliptic.P521().Params():
				err = verifyEcdsa(ecdsa.P521(), q, digest, signature)
			default:
				return nil, fmt.Errorf("unsupported curve: %v", curve)
			}

			if err == nil {
				return [][]byte{{1}}, nil
			}

			return [][]byte{{0}}, nil
		},
	}
}

func verifyEcdsa[P ecdsa.Point[P]](c *ecdsa.Curve[P], q []byte, digest []byte, sig *ecdsa.Signature) error {
	pub, err := ecdsa.NewPublicKey(c, q)
	if err != nil {
		return fmt.Errorf("invalid public key: %w", err)
	}

	return ecdsa.Verify(c, pub, digest, sig)
}

func lookupHash(name string) (func() fips140.Hash, error) {
	var h func() fips140.Hash

	switch name {
	case "SHA2-224":
		h = func() fips140.Hash { return sha256.New224() }
	case "SHA2-256":
		h = func() fips140.Hash { return sha256.New() }
	case "SHA2-384":
		h = func() fips140.Hash { return sha512.New384() }
	case "SHA2-512":
		h = func() fips140.Hash { return sha512.New() }
	case "SHA2-512/224":
		h = func() fips140.Hash { return sha512.New512_224() }
	case "SHA2-512/256":
		h = func() fips140.Hash { return sha512.New512_256() }
	case "SHA3-224":
		h = func() fips140.Hash { return sha3.New224() }
	case "SHA3-256":
		h = func() fips140.Hash { return sha3.New256() }
	case "SHA3-384":
		h = func() fips140.Hash { return sha3.New384() }
	case "SHA3-512":
		h = func() fips140.Hash { return sha3.New512() }
	default:
		return nil, fmt.Errorf("unknown hash name: %q", name)
	}

	return h, nil
}

func cmdMlKem768KeyGenAft() command {
	return command{
		requiredArgs: 1, // Seed
		handler: func(args [][]byte) ([][]byte, error) {
			seed := args[0]

			dk, err := mlkem.NewDecapsulationKey768(seed)
			if err != nil {
				return nil, fmt.Errorf("generating ML-KEM 768 decapsulation key: %w", err)
			}

			// Important: we must return the full encoding of dk, not the seed.
			return [][]byte{dk.EncapsulationKey().Bytes(), mlkem.TestingOnlyExpandedBytes768(dk)}, nil
		},
	}
}

func cmdMlKem768EncapAft() command {
	return command{
		requiredArgs: 2, // Public key, entropy
		handler: func(args [][]byte) ([][]byte, error) {
			pk := args[0]
			entropy := args[1]

			ek, err := mlkem.NewEncapsulationKey768(pk)
			if err != nil {
				return nil, fmt.Errorf("generating ML-KEM 768 encapsulation key: %w", err)
			}

			if len(entropy) != 32 {
				return nil, fmt.Errorf("wrong entropy length: got %d, want 32", len(entropy))
			}

			sharedKey, ct := ek.EncapsulateInternal((*[32]byte)(entropy[:32]))

			return [][]byte{ct, sharedKey}, nil
		},
	}
}

func cmdMlKem768DecapAft() command {
	return command{
		requiredArgs: 2, // Private key, ciphertext
		handler: func(args [][]byte) ([][]byte, error) {
			pk := args[0]
			ct := args[1]

			dk, err := mlkem.TestingOnlyNewDecapsulationKey768(pk)
			if err != nil {
				return nil, fmt.Errorf("generating ML-KEM 768 decapsulation key: %w", err)
			}

			sharedKey, err := dk.Decapsulate(ct)
			if err != nil {
				return nil, fmt.Errorf("decapsulating ML-KEM 768 ciphertext: %w", err)
			}

			return [][]byte{sharedKey}, nil
		},
	}
}

func cmdMlKem1024KeyGenAft() command {
	return command{
		requiredArgs: 1, // Seed
		handler: func(args [][]byte) ([][]byte, error) {
			seed := args[0]

			dk, err := mlkem.NewDecapsulationKey1024(seed)
			if err != nil {
				return nil, fmt.Errorf("generating ML-KEM 1024 decapsulation key: %w", err)
			}

			// Important: we must return the full encoding of dk, not the seed.
			return [][]byte{dk.EncapsulationKey().Bytes(), mlkem.TestingOnlyExpandedBytes1024(dk)}, nil
		},
	}
}

func cmdMlKem1024EncapAft() command {
	return command{
		requiredArgs: 2, // Public key, entropy
		handler: func(args [][]byte) ([][]byte, error) {
			pk := args[0]
			entropy := args[1]

			ek, err := mlkem.NewEncapsulationKey1024(pk)
			if err != nil {
				return nil, fmt.Errorf("generating ML-KEM 1024 encapsulation key: %w", err)
			}

			if len(entropy) != 32 {
				return nil, fmt.Errorf("wrong entropy length: got %d, want 32", len(entropy))
			}

			sharedKey, ct := ek.EncapsulateInternal((*[32]byte)(entropy[:32]))

			return [][]byte{ct, sharedKey}, nil
		},
	}
}

func cmdMlKem1024DecapAft() command {
	return command{
		requiredArgs: 2, // Private key, ciphertext
		handler: func(args [][]byte) ([][]byte, error) {
			pk := args[0]
			ct := args[1]

			dk, err := mlkem.TestingOnlyNewDecapsulationKey1024(pk)
			if err != nil {
				return nil, fmt.Errorf("generating ML-KEM 1024 decapsulation key: %w", err)
			}

			sharedKey, err := dk.Decapsulate(ct)
			if err != nil {
				return nil, fmt.Errorf("decapsulating ML-KEM 1024 ciphertext: %w", err)
			}

			return [][]byte{sharedKey}, nil
		},
	}
}

func cmdHmacDrbgAft(h func() fips140.Hash) command {
	return command{
		requiredArgs: 6, // Output length, entropy, personalization, ad1, ad2, nonce
		handler: func(args [][]byte) ([][]byte, error) {
			outLen := binary.LittleEndian.Uint32(args[0])
			entropy := args[1]
			personalization := args[2]
			ad1 := args[3]
			ad2 := args[4]
			nonce := args[5]

			// Our capabilities describe no additional data support.
			if len(ad1) != 0 || len(ad2) != 0 {
				return nil, errors.New("additional data not supported")
			}

			// Our capabilities describe no prediction resistance (requires reseed) and no reseed.
			// So the test procedure is:
			//   * Instantiate DRBG
			//   * Generate but don't output
			//   * Generate output
			//   * Uninstantiate
			// See Table 7 in draft-vassilev-acvp-drbg
			out := make([]byte, outLen)
			drbg := ecdsa.TestingOnlyNewDRBG(h, entropy, nonce, personalization)
			drbg.Generate(out)
			drbg.Generate(out)

			return [][]byte{out}, nil
		},
	}
}

func lookupCurve(name string) (elliptic.Curve, error) {
	var c elliptic.Curve

	switch name {
	case "P-224":
		c = elliptic.P224()
	case "P-256":
		c = elliptic.P256()
	case "P-384":
		c = elliptic.P384()
	case "P-521":
		c = elliptic.P521()
	default:
		return nil, fmt.Errorf("unknown curve name: %q", name)
	}

	return c, nil
}

func cmdAesCbc(direction aesDirection) command {
	return command{
		requiredArgs: 4, // Key, ciphertext or plaintext, IV, num iterations
		handler: func(args [][]byte) ([][]byte, error) {
			if direction != aesEncrypt && direction != aesDecrypt {
				panic("invalid AES direction")
			}

			key := args[0]
			input := args[1]
			iv := args[2]
			numIterations := binary.LittleEndian.Uint32(args[3])

			blockCipher, err := aes.New(key)
			if err != nil {
				return nil, fmt.Errorf("creating AES block cipher with key len %d: %w", len(key), err)
			}

			if len(input)%blockCipher.BlockSize() != 0 || len(input) == 0 {
				return nil, fmt.Errorf("invalid ciphertext/plaintext size %d: not a multiple of block size %d",
					len(input), blockCipher.BlockSize())
			}

			if blockCipher.BlockSize() != len(iv) {
				return nil, fmt.Errorf("invalid IV size: expected %d, got %d", blockCipher.BlockSize(), len(iv))
			}

			result := make([]byte, len(input))
			prevResult := make([]byte, len(input))
			prevInput := make([]byte, len(input))

			for i := uint32(0); i < numIterations; i++ {
				copy(prevResult, result)

				if i > 0 {
					if direction == aesEncrypt {
						copy(iv, result)
					} else {
						copy(iv, prevInput)
					}
				}

				if direction == aesEncrypt {
					cbcEnc := aes.NewCBCEncrypter(blockCipher, [16]byte(iv))
					cbcEnc.CryptBlocks(result, input)
				} else {
					cbcDec := aes.NewCBCDecrypter(blockCipher, [16]byte(iv))
					cbcDec.CryptBlocks(result, input)
				}

				if direction == aesDecrypt {
					copy(prevInput, input)
				}

				if i == 0 {
					copy(input, iv)
				} else {
					copy(input, prevResult)
				}
			}

			return [][]byte{result, prevResult}, nil
		},
	}
}

func cmdAesCtr(direction aesDirection) command {
	return command{
		requiredArgs: 4, // Key, ciphertext or plaintext, initial counter, num iterations (constant 1)
		handler: func(args [][]byte) ([][]byte, error) {
			if direction != aesEncrypt && direction != aesDecrypt {
				panic("invalid AES direction")
			}

			key := args[0]
			input := args[1]
			iv := args[2]
			numIterations := binary.LittleEndian.Uint32(args[3])
			if numIterations != 1 {
				return nil, fmt.Errorf("invalid num iterations: expected 1, got %d", numIterations)
			}

			if len(iv) != aes.BlockSize {
				return nil, fmt.Errorf("invalid IV size: expected %d, got %d", aes.BlockSize, len(iv))
			}

			blockCipher, err := aes.New(key)
			if err != nil {
				return nil, fmt.Errorf("creating AES block cipher with key len %d: %w", len(key), err)
			}

			result := make([]byte, len(input))
			stream := aes.NewCTR(blockCipher, iv)
			stream.XORKeyStream(result, input)

			return [][]byte{result}, nil
		},
	}
}

func cmdAesGcmSeal(randNonce bool) command {
	return command{
		requiredArgs: 5, // tag len, key, plaintext, nonce (empty for randNonce), additional data
		handler: func(args [][]byte) ([][]byte, error) {
			tagLen := binary.LittleEndian.Uint32(args[0])
			key := args[1]
			plaintext := args[2]
			nonce := args[3]
			additionalData := args[4]

			blockCipher, err := aes.New(key)
			if err != nil {
				return nil, fmt.Errorf("creating AES block cipher with key len %d: %w", len(key), err)
			}

			aesGCM, err := gcm.New(blockCipher, 12, int(tagLen))
			if err != nil {
				return nil, fmt.Errorf("creating AES-GCM with tag len %d: %w", tagLen, err)
			}

			var ct []byte
			if !randNonce {
				ct = aesGCM.Seal(nil, nonce, plaintext, additionalData)
			} else {
				var internalNonce [12]byte
				ct = make([]byte, len(plaintext)+16)
				gcm.SealWithRandomNonce(aesGCM, internalNonce[:], ct, plaintext, additionalData)
				// acvptool expects the internally generated nonce to be appended to the end of the ciphertext.
				ct = append(ct, internalNonce[:]...)
			}

			return [][]byte{ct}, nil
		},
	}
}

func cmdAesGcmOpen(randNonce bool) command {
	return command{
		requiredArgs: 5, // tag len, key, ciphertext, nonce (empty for randNonce), additional data
		handler: func(args [][]byte) ([][]byte, error) {

			tagLen := binary.LittleEndian.Uint32(args[0])
			key := args[1]
			ciphertext := args[2]
			nonce := args[3]
			additionalData := args[4]

			blockCipher, err := aes.New(key)
			if err != nil {
				return nil, fmt.Errorf("creating AES block cipher with key len %d: %w", len(key), err)
			}

			aesGCM, err := gcm.New(blockCipher, 12, int(tagLen))
			if err != nil {
				return nil, fmt.Errorf("creating AES-GCM with tag len %d: %w", tagLen, err)
			}

			if randNonce {
				// for randNonce tests acvptool appends the nonce to the end of the ciphertext.
				nonce = ciphertext[len(ciphertext)-12:]
				ciphertext = ciphertext[:len(ciphertext)-12]
			}

			pt, err := aesGCM.Open(nil, nonce, ciphertext, additionalData)
			if err != nil {
				return [][]byte{{0}, nil}, nil
			}

			return [][]byte{{1}, pt}, nil
		},
	}
}

func cmdCmacAesAft() command {
	return command{
		requiredArgs: 3, // Number of output bytes, key, message
		handler: func(args [][]byte) ([][]byte, error) {
			// safe to truncate to int based on our capabilities describing a max MAC output len of 128 bits.
			outputLen := int(binary.LittleEndian.Uint32(args[0]))
			key := args[1]
			message := args[2]

			blockCipher, err := aes.New(key)
			if err != nil {
				return nil, fmt.Errorf("creating AES block cipher with key len %d: %w", len(key), err)
			}

			cmac := gcm.NewCMAC(blockCipher)
			tag := cmac.MAC(message)

			if outputLen > len(tag) {
				return nil, fmt.Errorf("invalid output length: expected %d, got %d", outputLen, len(tag))
			}

			return [][]byte{tag[:outputLen]}, nil
		},
	}
}

func cmdCmacAesVerifyAft() command {
	return command{
		requiredArgs: 3, // Key, message, claimed MAC
		handler: func(args [][]byte) ([][]byte, error) {
			key := args[0]
			message := args[1]
			claimedMAC := args[2]

			blockCipher, err := aes.New(key)
			if err != nil {
				return nil, fmt.Errorf("creating AES block cipher with key len %d: %w", len(key), err)
			}

			cmac := gcm.NewCMAC(blockCipher)
			tag := cmac.MAC(message)

			if subtle.ConstantTimeCompare(tag[:len(claimedMAC)], claimedMAC) != 1 {
				return [][]byte{{0}}, nil
			}

			return [][]byte{{1}}, nil
		},
	}
}

func TestACVP(t *testing.T) {
	testenv.SkipIfShortAndSlow(t)

	const (
		bsslModule    = "boringssl.googlesource.com/boringssl.git"
		bsslVersion   = "v0.0.0-20250108043213-d3f61eeacbf7"
		goAcvpModule  = "github.com/cpu/go-acvp"
		goAcvpVersion = "v0.0.0-20250102201911-6839fc40f9f8"
	)

	// In crypto/tls/bogo_shim_test.go the test is skipped if run on a builder with runtime.GOOS == "windows"
	// due to flaky networking. It may be necessary to do the same here.

	// Stat the acvp test config file so the test will be re-run if it changes, invalidating cached results
	// from the old config.
	if _, err := os.Stat("acvp_test.config.json"); err != nil {
		t.Fatalf("failed to stat config file: %s", err)
	}

	// Fetch the BSSL module and use the JSON output to find the absolute path to the dir.
	bsslDir := cryptotest.FetchModule(t, bsslModule, bsslVersion)

	t.Log("building acvptool")

	// Build the acvptool binary.
	toolPath := filepath.Join(t.TempDir(), "acvptool.exe")
	goTool := testenv.GoToolPath(t)
	cmd := testenv.Command(t, goTool,
		"build",
		"-o", toolPath,
		"./util/fipstools/acvp/acvptool")
	cmd.Dir = bsslDir
	out := &strings.Builder{}
	cmd.Stderr = out
	if err := cmd.Run(); err != nil {
		t.Fatalf("failed to build acvptool: %s\n%s", err, out.String())
	}

	// Similarly, fetch the ACVP data module that has vectors/expected answers.
	dataDir := cryptotest.FetchModule(t, goAcvpModule, goAcvpVersion)

	cwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("failed to fetch cwd: %s", err)
	}
	configPath := filepath.Join(cwd, "acvp_test.config.json")
	t.Logf("running check_expected.go\ncwd: %q\ndata_dir: %q\nconfig: %q\ntool: %q\nmodule-wrapper: %q\n",
		cwd, dataDir, configPath, toolPath, os.Args[0])

	// Run the check_expected test driver using the acvptool we built, and this test binary as the
	// module wrapper. The file paths in the config file are specified relative to the dataDir root
	// so we run the command from that dir.
	args := []string{
		"run",
		filepath.Join(bsslDir, "util/fipstools/acvp/acvptool/test/check_expected.go"),
		"-tool",
		toolPath,
		// Note: module prefix must match Wrapper value in acvp_test.config.json.
		"-module-wrappers", "go:" + os.Args[0],
		"-tests", configPath,
	}
	cmd = testenv.Command(t, goTool, args...)
	cmd.Dir = dataDir
	cmd.Env = append(os.Environ(), "ACVP_WRAPPER=1")
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("failed to run acvp tests: %s\n%s", err, string(output))
	}
	t.Log(string(output))
}

func TestTooFewArgs(t *testing.T) {
	commands["test"] = command{
		requiredArgs: 1,
		handler: func(args [][]byte) ([][]byte, error) {
			if gotArgs := len(args); gotArgs != 1 {
				return nil, fmt.Errorf("expected 1 args, got %d", gotArgs)
			}
			return nil, nil
		},
	}

	var output bytes.Buffer
	err := processingLoop(mockRequest(t, "test", nil), &output)
	if err == nil {
		t.Fatalf("expected error, got nil")
	}
	expectedErr := "expected 1 args, got 0"
	if !strings.Contains(err.Error(), expectedErr) {
		t.Errorf("expected error to contain %q, got %v", expectedErr, err)
	}
}

func TestTooManyArgs(t *testing.T) {
	commands["test"] = command{
		requiredArgs: 1,
		handler: func(args [][]byte) ([][]byte, error) {
			if gotArgs := len(args); gotArgs != 1 {
				return nil, fmt.Errorf("expected 1 args, got %d", gotArgs)
			}
			return nil, nil
		},
	}

	var output bytes.Buffer
	err := processingLoop(mockRequest(
		t, "test", [][]byte{[]byte("one"), []byte("two")}), &output)
	if err == nil {
		t.Fatalf("expected error, got nil")
	}
	expectedErr := "expected 1 args, got 2"
	if !strings.Contains(err.Error(), expectedErr) {
		t.Errorf("expected error to contain %q, got %v", expectedErr, err)
	}
}

func TestGetConfig(t *testing.T) {
	var output bytes.Buffer
	err := processingLoop(mockRequest(t, "getConfig", nil), &output)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	respArgs := readResponse(t, &output)
	if len(respArgs) != 1 {
		t.Fatalf("expected 1 response arg, got %d", len(respArgs))
	}

	if !bytes.Equal(respArgs[0], capabilitiesJson) {
		t.Errorf("expected config %q, got %q", string(capabilitiesJson), string(respArgs[0]))
	}
}

func TestSha2256(t *testing.T) {
	testMessage := []byte("gophers eat grass")
	expectedDigest := []byte{
		188, 142, 10, 214, 48, 236, 72, 143, 70, 216, 223, 205, 219, 69, 53, 29,
		205, 207, 162, 6, 14, 70, 113, 60, 251, 170, 201, 236, 119, 39, 141, 172,
	}

	var output bytes.Buffer
	err := processingLoop(mockRequest(t, "SHA2-256", [][]byte{testMessage}), &output)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	respArgs := readResponse(t, &output)
	if len(respArgs) != 1 {
		t.Fatalf("expected 1 response arg, got %d", len(respArgs))
	}

	if !bytes.Equal(respArgs[0], expectedDigest) {
		t.Errorf("expected digest %v, got %v", expectedDigest, respArgs[0])
	}
}

func mockRequest(t *testing.T, cmd string, args [][]byte) io.Reader {
	t.Helper()

	msgData := append([][]byte{[]byte(cmd)}, args...)

	var buf bytes.Buffer
	if err := writeResponse(&buf, msgData); err != nil {
		t.Fatalf("writeResponse error: %v", err)
	}

	return &buf
}

func readResponse(t *testing.T, reader io.Reader) [][]byte {
	var numArgs uint32
	if err := binary.Read(reader, binary.LittleEndian, &numArgs); err != nil {
		t.Fatalf("failed to read response args count: %v", err)
	}

	args, err := readArgs(reader, numArgs)
	if err != nil {
		t.Fatalf("failed to read %d response args: %v", numArgs, err)
	}

	return args
}
