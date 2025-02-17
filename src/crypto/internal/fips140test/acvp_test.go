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
	"crypto/internal/fips140/bigmod"
	"crypto/internal/fips140/drbg"
	"crypto/internal/fips140/ecdh"
	"crypto/internal/fips140/ecdsa"
	"crypto/internal/fips140/ed25519"
	"crypto/internal/fips140/edwards25519"
	"crypto/internal/fips140/hkdf"
	"crypto/internal/fips140/hmac"
	"crypto/internal/fips140/mlkem"
	"crypto/internal/fips140/pbkdf2"
	"crypto/internal/fips140/rsa"
	"crypto/internal/fips140/sha256"
	"crypto/internal/fips140/sha3"
	"crypto/internal/fips140/sha512"
	"crypto/internal/fips140/ssh"
	"crypto/internal/fips140/subtle"
	"crypto/internal/fips140/tls12"
	"crypto/internal/fips140/tls13"
	"crypto/internal/impl"
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

var noPAAPAI = os.Getenv("GONOPAAPAI") == "1"

func TestMain(m *testing.M) {
	if noPAAPAI {
		for _, p := range impl.Packages() {
			impl.Select(p, "")
		}
	}
	if os.Getenv("ACVP_WRAPPER") == "1" {
		wrapperMain()
	} else {
		os.Exit(m.Run())
	}
}

func wrapperMain() {
	if !fips140.Enabled {
		fmt.Fprintln(os.Stderr, "ACVP wrapper must be run with GODEBUG=fips140=on")
		os.Exit(2)
	}
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
	// SHA3 and SHAKE algorithm capabilities:
	//   https://pages.nist.gov/ACVP/draft-celi-acvp-sha3.html#name-sha3-and-shake-algorithm-ca
	// cSHAKE algorithm capabilities:
	//   https://pages.nist.gov/ACVP/draft-celi-acvp-xof.html#section-7.2
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
	// HKDF KDA algorithm capabilities:
	//   https://pages.nist.gov/ACVP/draft-hammett-acvp-kas-kdf-hkdf.html#section-7.3
	// OneStepNoCounter KDA algorithm capabilities:
	//   https://pages.nist.gov/ACVP/draft-hammett-acvp-kas-kdf-onestepnocounter.html#section-7.2
	// TLS 1.2 KDF algorithm capabilities:
	//   https://pages.nist.gov/ACVP/draft-celi-acvp-kdf-tls.html#section-7.2
	// TLS 1.3 KDF algorithm capabilities:
	//   https://pages.nist.gov/ACVP/draft-hammett-acvp-kdf-tls-v1.3.html#section-7.2
	// SSH KDF algorithm capabilities:
	//   https://pages.nist.gov/ACVP/draft-celi-acvp-kdf-ssh.html#section-7.2
	// ECDH algorithm capabilities:
	//   https://pages.nist.gov/ACVP/draft-hammett-acvp-kas-ssc-ecc.html#section-7.3
	// HMAC DRBG and CTR DRBG algorithm capabilities:
	//   https://pages.nist.gov/ACVP/draft-vassilev-acvp-drbg.html#section-7.2
	// KDF-Counter and KDF-Feedback algorithm capabilities:
	//   https://pages.nist.gov/ACVP/draft-celi-acvp-kbkdf.html#section-7.3
	// RSA algorithm capabilities:
	//   https://pages.nist.gov/ACVP/draft-celi-acvp-rsa.html#section-7.3
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

		// Note: SHAKE AFT and VOT test types can be handled by the same command
		// handler impl, but use distinct acvptool command names, and so are
		// registered twice with the same digest: once under "SHAKE-xxx" for AFT,
		// and once under"SHAKE-xxx/VOT" for VOT.
		"SHAKE-128":     cmdShakeAftVot(sha3.NewShake128()),
		"SHAKE-128/VOT": cmdShakeAftVot(sha3.NewShake128()),
		"SHAKE-128/MCT": cmdShakeMct(sha3.NewShake128()),
		"SHAKE-256":     cmdShakeAftVot(sha3.NewShake256()),
		"SHAKE-256/VOT": cmdShakeAftVot(sha3.NewShake256()),
		"SHAKE-256/MCT": cmdShakeMct(sha3.NewShake256()),

		"cSHAKE-128":     cmdCShakeAft(func(N, S []byte) *sha3.SHAKE { return sha3.NewCShake128(N, S) }),
		"cSHAKE-128/MCT": cmdCShakeMct(func(N, S []byte) *sha3.SHAKE { return sha3.NewCShake128(N, S) }),
		"cSHAKE-256":     cmdCShakeAft(func(N, S []byte) *sha3.SHAKE { return sha3.NewCShake256(N, S) }),
		"cSHAKE-256/MCT": cmdCShakeMct(func(N, S []byte) *sha3.SHAKE { return sha3.NewCShake256(N, S) }),

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

		"HKDF/SHA2-224":     cmdHkdfAft(func() fips140.Hash { return sha256.New224() }),
		"HKDF/SHA2-256":     cmdHkdfAft(func() fips140.Hash { return sha256.New() }),
		"HKDF/SHA2-384":     cmdHkdfAft(func() fips140.Hash { return sha512.New384() }),
		"HKDF/SHA2-512":     cmdHkdfAft(func() fips140.Hash { return sha512.New() }),
		"HKDF/SHA2-512/224": cmdHkdfAft(func() fips140.Hash { return sha512.New512_224() }),
		"HKDF/SHA2-512/256": cmdHkdfAft(func() fips140.Hash { return sha512.New512_256() }),
		"HKDF/SHA3-224":     cmdHkdfAft(func() fips140.Hash { return sha3.New224() }),
		"HKDF/SHA3-256":     cmdHkdfAft(func() fips140.Hash { return sha3.New256() }),
		"HKDF/SHA3-384":     cmdHkdfAft(func() fips140.Hash { return sha3.New384() }),
		"HKDF/SHA3-512":     cmdHkdfAft(func() fips140.Hash { return sha3.New512() }),

		"HKDFExtract/SHA2-256":     cmdHkdfExtractAft(func() fips140.Hash { return sha256.New() }),
		"HKDFExtract/SHA2-384":     cmdHkdfExtractAft(func() fips140.Hash { return sha512.New384() }),
		"HKDFExpandLabel/SHA2-256": cmdHkdfExpandLabelAft(func() fips140.Hash { return sha256.New() }),
		"HKDFExpandLabel/SHA2-384": cmdHkdfExpandLabelAft(func() fips140.Hash { return sha512.New384() }),

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

		// Note: Only SHA2-256, SHA2-384 and SHA2-512 are valid hash functions for TLSKDF.
		// 		 See https://pages.nist.gov/ACVP/draft-celi-acvp-kdf-tls.html#section-7.2.1
		"TLSKDF/1.2/SHA2-256": cmdTlsKdf12Aft(func() fips140.Hash { return sha256.New() }),
		"TLSKDF/1.2/SHA2-384": cmdTlsKdf12Aft(func() fips140.Hash { return sha512.New384() }),
		"TLSKDF/1.2/SHA2-512": cmdTlsKdf12Aft(func() fips140.Hash { return sha512.New() }),

		// Note: only SHA2-224, SHA2-256, SHA2-384 and SHA2-512 are valid hash functions for SSHKDF.
		// 		 See https://pages.nist.gov/ACVP/draft-celi-acvp-kdf-ssh.html#section-7.2.1
		"SSHKDF/SHA2-224/client": cmdSshKdfAft(func() fips140.Hash { return sha256.New224() }, ssh.ClientKeys),
		"SSHKDF/SHA2-224/server": cmdSshKdfAft(func() fips140.Hash { return sha256.New224() }, ssh.ServerKeys),
		"SSHKDF/SHA2-256/client": cmdSshKdfAft(func() fips140.Hash { return sha256.New() }, ssh.ClientKeys),
		"SSHKDF/SHA2-256/server": cmdSshKdfAft(func() fips140.Hash { return sha256.New() }, ssh.ServerKeys),
		"SSHKDF/SHA2-384/client": cmdSshKdfAft(func() fips140.Hash { return sha512.New384() }, ssh.ClientKeys),
		"SSHKDF/SHA2-384/server": cmdSshKdfAft(func() fips140.Hash { return sha512.New384() }, ssh.ServerKeys),
		"SSHKDF/SHA2-512/client": cmdSshKdfAft(func() fips140.Hash { return sha512.New() }, ssh.ClientKeys),
		"SSHKDF/SHA2-512/server": cmdSshKdfAft(func() fips140.Hash { return sha512.New() }, ssh.ServerKeys),

		"ECDH/P-224": cmdEcdhAftVal(ecdh.P224()),
		"ECDH/P-256": cmdEcdhAftVal(ecdh.P256()),
		"ECDH/P-384": cmdEcdhAftVal(ecdh.P384()),
		"ECDH/P-521": cmdEcdhAftVal(ecdh.P521()),

		"ctrDRBG/AES-256":        cmdCtrDrbgAft(),
		"ctrDRBG-reseed/AES-256": cmdCtrDrbgReseedAft(),

		"RSA/keyGen": cmdRsaKeyGenAft(),

		"RSA/sigGen/SHA2-224/pkcs1v1.5": cmdRsaSigGenAft(func() fips140.Hash { return sha256.New224() }, "SHA-224", false),
		"RSA/sigGen/SHA2-256/pkcs1v1.5": cmdRsaSigGenAft(func() fips140.Hash { return sha256.New() }, "SHA-256", false),
		"RSA/sigGen/SHA2-384/pkcs1v1.5": cmdRsaSigGenAft(func() fips140.Hash { return sha512.New384() }, "SHA-384", false),
		"RSA/sigGen/SHA2-512/pkcs1v1.5": cmdRsaSigGenAft(func() fips140.Hash { return sha512.New() }, "SHA-512", false),
		"RSA/sigGen/SHA2-224/pss":       cmdRsaSigGenAft(func() fips140.Hash { return sha256.New224() }, "SHA-224", true),
		"RSA/sigGen/SHA2-256/pss":       cmdRsaSigGenAft(func() fips140.Hash { return sha256.New() }, "SHA-256", true),
		"RSA/sigGen/SHA2-384/pss":       cmdRsaSigGenAft(func() fips140.Hash { return sha512.New384() }, "SHA-384", true),
		"RSA/sigGen/SHA2-512/pss":       cmdRsaSigGenAft(func() fips140.Hash { return sha512.New() }, "SHA-512", true),

		"RSA/sigVer/SHA2-224/pkcs1v1.5": cmdRsaSigVerAft(func() fips140.Hash { return sha256.New224() }, "SHA-224", false),
		"RSA/sigVer/SHA2-256/pkcs1v1.5": cmdRsaSigVerAft(func() fips140.Hash { return sha256.New() }, "SHA-256", false),
		"RSA/sigVer/SHA2-384/pkcs1v1.5": cmdRsaSigVerAft(func() fips140.Hash { return sha512.New384() }, "SHA-384", false),
		"RSA/sigVer/SHA2-512/pkcs1v1.5": cmdRsaSigVerAft(func() fips140.Hash { return sha512.New() }, "SHA-512", false),
		"RSA/sigVer/SHA2-224/pss":       cmdRsaSigVerAft(func() fips140.Hash { return sha256.New224() }, "SHA-224", true),
		"RSA/sigVer/SHA2-256/pss":       cmdRsaSigVerAft(func() fips140.Hash { return sha256.New() }, "SHA-256", true),
		"RSA/sigVer/SHA2-384/pss":       cmdRsaSigVerAft(func() fips140.Hash { return sha512.New384() }, "SHA-384", true),
		"RSA/sigVer/SHA2-512/pss":       cmdRsaSigVerAft(func() fips140.Hash { return sha512.New() }, "SHA-512", true),

		"KDF-counter":  cmdKdfCounterAft(),
		"KDF-feedback": cmdKdfFeedbackAft(),

		"OneStepNoCounter/HMAC-SHA2-224":     cmdOneStepNoCounterHmacAft(func() fips140.Hash { return sha256.New224() }),
		"OneStepNoCounter/HMAC-SHA2-256":     cmdOneStepNoCounterHmacAft(func() fips140.Hash { return sha256.New() }),
		"OneStepNoCounter/HMAC-SHA2-384":     cmdOneStepNoCounterHmacAft(func() fips140.Hash { return sha512.New384() }),
		"OneStepNoCounter/HMAC-SHA2-512":     cmdOneStepNoCounterHmacAft(func() fips140.Hash { return sha512.New() }),
		"OneStepNoCounter/HMAC-SHA2-512/224": cmdOneStepNoCounterHmacAft(func() fips140.Hash { return sha512.New512_224() }),
		"OneStepNoCounter/HMAC-SHA2-512/256": cmdOneStepNoCounterHmacAft(func() fips140.Hash { return sha512.New512_256() }),
		"OneStepNoCounter/HMAC-SHA3-224":     cmdOneStepNoCounterHmacAft(func() fips140.Hash { return sha3.New224() }),
		"OneStepNoCounter/HMAC-SHA3-256":     cmdOneStepNoCounterHmacAft(func() fips140.Hash { return sha3.New256() }),
		"OneStepNoCounter/HMAC-SHA3-384":     cmdOneStepNoCounterHmacAft(func() fips140.Hash { return sha3.New384() }),
		"OneStepNoCounter/HMAC-SHA3-512":     cmdOneStepNoCounterHmacAft(func() fips140.Hash { return sha3.New512() }),

		"KTS-IFC/SHA2-224/initiator":     cmdKtsIfcInitiatorAft(func() fips140.Hash { return sha256.New224() }),
		"KTS-IFC/SHA2-224/responder":     cmdKtsIfcResponderAft(func() fips140.Hash { return sha256.New224() }),
		"KTS-IFC/SHA2-256/initiator":     cmdKtsIfcInitiatorAft(func() fips140.Hash { return sha256.New() }),
		"KTS-IFC/SHA2-256/responder":     cmdKtsIfcResponderAft(func() fips140.Hash { return sha256.New() }),
		"KTS-IFC/SHA2-384/initiator":     cmdKtsIfcInitiatorAft(func() fips140.Hash { return sha512.New384() }),
		"KTS-IFC/SHA2-384/responder":     cmdKtsIfcResponderAft(func() fips140.Hash { return sha512.New384() }),
		"KTS-IFC/SHA2-512/initiator":     cmdKtsIfcInitiatorAft(func() fips140.Hash { return sha512.New() }),
		"KTS-IFC/SHA2-512/responder":     cmdKtsIfcResponderAft(func() fips140.Hash { return sha512.New() }),
		"KTS-IFC/SHA2-512/224/initiator": cmdKtsIfcInitiatorAft(func() fips140.Hash { return sha512.New512_224() }),
		"KTS-IFC/SHA2-512/224/responder": cmdKtsIfcResponderAft(func() fips140.Hash { return sha512.New512_224() }),
		"KTS-IFC/SHA2-512/256/initiator": cmdKtsIfcInitiatorAft(func() fips140.Hash { return sha512.New512_256() }),
		"KTS-IFC/SHA2-512/256/responder": cmdKtsIfcResponderAft(func() fips140.Hash { return sha512.New512_256() }),
		"KTS-IFC/SHA3-224/initiator":     cmdKtsIfcInitiatorAft(func() fips140.Hash { return sha3.New224() }),
		"KTS-IFC/SHA3-224/responder":     cmdKtsIfcResponderAft(func() fips140.Hash { return sha3.New224() }),
		"KTS-IFC/SHA3-256/initiator":     cmdKtsIfcInitiatorAft(func() fips140.Hash { return sha3.New256() }),
		"KTS-IFC/SHA3-256/responder":     cmdKtsIfcResponderAft(func() fips140.Hash { return sha3.New256() }),
		"KTS-IFC/SHA3-384/initiator":     cmdKtsIfcInitiatorAft(func() fips140.Hash { return sha3.New384() }),
		"KTS-IFC/SHA3-384/responder":     cmdKtsIfcResponderAft(func() fips140.Hash { return sha3.New384() }),
		"KTS-IFC/SHA3-512/initiator":     cmdKtsIfcInitiatorAft(func() fips140.Hash { return sha3.New512() }),
		"KTS-IFC/SHA3-512/responder":     cmdKtsIfcResponderAft(func() fips140.Hash { return sha3.New512() }),
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

func cmdShakeAftVot(h *sha3.SHAKE) command {
	return command{
		requiredArgs: 2, // Message, output length (bytes)
		handler: func(args [][]byte) ([][]byte, error) {
			msg := args[0]

			outLenBytes := binary.LittleEndian.Uint32(args[1])
			digest := make([]byte, outLenBytes)

			h.Reset()
			h.Write(msg)
			h.Read(digest)

			return [][]byte{digest}, nil
		},
	}
}

func cmdShakeMct(h *sha3.SHAKE) command {
	return command{
		requiredArgs: 4, // Seed message, min output length (bytes), max output length (bytes), output length (bytes)
		handler: func(args [][]byte) ([][]byte, error) {
			md := args[0]
			minOutBytes := binary.LittleEndian.Uint32(args[1])
			maxOutBytes := binary.LittleEndian.Uint32(args[2])

			outputLenBytes := binary.LittleEndian.Uint32(args[3])
			if outputLenBytes < 2 {
				return nil, fmt.Errorf("invalid output length: %d", outputLenBytes)
			}

			rangeBytes := maxOutBytes - minOutBytes + 1
			if rangeBytes == 0 {
				return nil, fmt.Errorf("invalid maxOutBytes and minOutBytes: %d, %d", maxOutBytes, minOutBytes)
			}

			for i := 0; i < 1000; i++ {
				// "The MSG[i] input to SHAKE MUST always contain at least 128 bits. If this is not the case
				// as the previous digest was too short, append empty bits to the rightmost side of the digest."
				boundary := min(len(md), 16)
				msg := make([]byte, 16)
				copy(msg, md[:boundary])

				//  MD[i] = SHAKE(MSG[i], OutputLen * 8)
				h.Reset()
				h.Write(msg)
				digest := make([]byte, outputLenBytes)
				h.Read(digest)
				md = digest

				// RightmostOutputBits = 16 rightmost bits of MD[i] as an integer
				// OutputLen = minOutBytes + (RightmostOutputBits % Range)
				rightmostOutput := uint32(md[outputLenBytes-2])<<8 | uint32(md[outputLenBytes-1])
				outputLenBytes = minOutBytes + (rightmostOutput % rangeBytes)
			}

			encodedOutputLenBytes := make([]byte, 4)
			binary.LittleEndian.PutUint32(encodedOutputLenBytes, outputLenBytes)

			return [][]byte{md, encodedOutputLenBytes}, nil
		},
	}
}

func cmdCShakeAft(hFn func(N, S []byte) *sha3.SHAKE) command {
	return command{
		requiredArgs: 4, // Message, output length bytes, function name, customization
		handler: func(args [][]byte) ([][]byte, error) {
			msg := args[0]
			outLenBytes := binary.LittleEndian.Uint32(args[1])
			functionName := args[2]
			customization := args[3]

			h := hFn(functionName, customization)
			h.Write(msg)

			out := make([]byte, outLenBytes)
			h.Read(out)

			return [][]byte{out}, nil
		},
	}
}

func cmdCShakeMct(hFn func(N, S []byte) *sha3.SHAKE) command {
	return command{
		requiredArgs: 6, // Message, min output length (bits), max output length (bits), output length (bits), increment (bits), customization
		handler: func(args [][]byte) ([][]byte, error) {
			message := args[0]
			minOutLenBytes := binary.LittleEndian.Uint32(args[1])
			maxOutLenBytes := binary.LittleEndian.Uint32(args[2])
			outputLenBytes := binary.LittleEndian.Uint32(args[3])
			incrementBytes := binary.LittleEndian.Uint32(args[4])
			customization := args[5]

			if outputLenBytes < 2 {
				return nil, fmt.Errorf("invalid output length: %d", outputLenBytes)
			}

			rangeBits := (maxOutLenBytes*8 - minOutLenBytes*8) + 1
			if rangeBits == 0 {
				return nil, fmt.Errorf("invalid maxOutLenBytes and minOutLenBytes: %d, %d", maxOutLenBytes, minOutLenBytes)
			}

			// cSHAKE Monte Carlo test inner loop:
			//   https://pages.nist.gov/ACVP/draft-celi-acvp-xof.html#section-6.2.1
			for i := 0; i < 1000; i++ {
				// InnerMsg = Left(Output[i-1] || ZeroBits(128), 128);
				boundary := min(len(message), 16)
				innerMsg := make([]byte, 16)
				copy(innerMsg, message[:boundary])

				// Output[i] = CSHAKE(InnerMsg, OutputLen, FunctionName, Customization);
				h := hFn(nil, customization) // Note: function name fixed to "" for MCT.
				h.Write(innerMsg)
				digest := make([]byte, outputLenBytes)
				h.Read(digest)
				message = digest

				// Rightmost_Output_bits = Right(Output[i], 16);
				rightmostOutput := digest[outputLenBytes-2:]
				// IMPORTANT: the specification says:
				//   NOTE: For the "Rightmost_Output_bits % Range" operation, the Rightmost_Output_bits bit string
				//   should be interpretted as a little endian-encoded number.
				// This is **a lie**! It has to be interpreted as a big-endian number.
				rightmostOutputBE := binary.BigEndian.Uint16(rightmostOutput)

				// OutputLen = MinOutLen + (floor((Rightmost_Output_bits % Range) / OutLenIncrement) * OutLenIncrement);
				incrementBits := incrementBytes * 8
				outputLenBits := (minOutLenBytes * 8) + (((uint32)(rightmostOutputBE)%rangeBits)/incrementBits)*incrementBits
				outputLenBytes = outputLenBits / 8

				// Customization = BitsToString(InnerMsg || Rightmost_Output_bits);
				msgWithBits := append(innerMsg, rightmostOutput...)
				customization = make([]byte, len(msgWithBits))
				for i, b := range msgWithBits {
					customization[i] = (b % 26) + 65
				}
			}

			encodedOutputLenBytes := make([]byte, 4)
			binary.LittleEndian.PutUint32(encodedOutputLenBytes, outputLenBytes)

			return [][]byte{message, encodedOutputLenBytes, customization}, nil
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

func cmdHkdfAft(h func() fips140.Hash) command {
	return command{
		requiredArgs: 4, // Key, salt, info, length bytes
		handler: func(args [][]byte) ([][]byte, error) {
			key := args[0]
			salt := args[1]
			info := args[2]
			keyLen := int(binary.LittleEndian.Uint32(args[3]))

			return [][]byte{hkdf.Key(h, key, salt, string(info), keyLen)}, nil
		},
	}
}

func cmdHkdfExtractAft(h func() fips140.Hash) command {
	return command{
		requiredArgs: 2, // secret, salt
		handler: func(args [][]byte) ([][]byte, error) {
			secret := args[0]
			salt := args[1]

			return [][]byte{hkdf.Extract(h, secret, salt)}, nil
		},
	}
}

func cmdHkdfExpandLabelAft(h func() fips140.Hash) command {
	return command{
		requiredArgs: 4, // output length, secret, label, transcript hash
		handler: func(args [][]byte) ([][]byte, error) {
			keyLen := int(binary.LittleEndian.Uint32(args[0]))
			secret := args[1]
			label := args[2]
			transcriptHash := args[3]

			return [][]byte{tls13.ExpandLabel(h, secret, string(label), transcriptHash, keyLen)}, nil
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

func cmdTlsKdf12Aft(h func() fips140.Hash) command {
	return command{
		requiredArgs: 5, // Number output bytes, secret, label, seed1, seed2
		handler: func(args [][]byte) ([][]byte, error) {
			outputLen := binary.LittleEndian.Uint32(args[0])
			secret := args[1]
			label := string(args[2])
			seed1 := args[3]
			seed2 := args[4]

			return [][]byte{tls12.PRF(h, secret, label, append(seed1, seed2...), int(outputLen))}, nil
		},
	}
}

func cmdSshKdfAft(hFunc func() fips140.Hash, direction ssh.Direction) command {
	return command{
		requiredArgs: 4, // K, H, SessionID, cipher
		handler: func(args [][]byte) ([][]byte, error) {
			k := args[0]
			h := args[1]
			sessionID := args[2]
			cipher := string(args[3])

			var keyLen int
			switch cipher {
			case "AES-128":
				keyLen = 16
			case "AES-192":
				keyLen = 24
			case "AES-256":
				keyLen = 32
			default:
				return nil, fmt.Errorf("unsupported cipher: %q", cipher)
			}

			ivKey, encKey, intKey := ssh.Keys(hFunc, direction, k, h, sessionID, 16, keyLen, hFunc().Size())
			return [][]byte{ivKey, encKey, intKey}, nil
		},
	}
}

func cmdEcdhAftVal[P ecdh.Point[P]](curve *ecdh.Curve[P]) command {
	return command{
		requiredArgs: 3, // X, Y, private key (empty for Val type tests)
		handler: func(args [][]byte) ([][]byte, error) {
			peerX := args[0]
			peerY := args[1]
			rawSk := args[2]

			uncompressedPk := append([]byte{4}, append(peerX, peerY...)...) // 4 for uncompressed point format
			pk, err := ecdh.NewPublicKey(curve, uncompressedPk)
			if err != nil {
				return nil, fmt.Errorf("invalid peer public key x,y: %v", err)
			}

			var sk *ecdh.PrivateKey
			if len(rawSk) > 0 {
				sk, err = ecdh.NewPrivateKey(curve, rawSk)
			} else {
				sk, err = ecdh.GenerateKey(curve, rand.Reader)
			}
			if err != nil {
				return nil, fmt.Errorf("private key error: %v", err)
			}

			pubBytes := sk.PublicKey().Bytes()
			coordLen := (len(pubBytes) - 1) / 2
			x := pubBytes[1 : 1+coordLen]
			y := pubBytes[1+coordLen:]

			secret, err := ecdh.ECDH(curve, sk, pk)
			if err != nil {
				return nil, fmt.Errorf("key agreement failed: %v", err)
			}

			return [][]byte{x, y, secret}, nil
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

func cmdCtrDrbgAft() command {
	return command{
		requiredArgs: 6, // Output length, entropy, personalization, ad1, ad2, nonce
		handler: func(args [][]byte) ([][]byte, error) {
			return acvpCtrDrbg{
				outLen:          binary.LittleEndian.Uint32(args[0]),
				entropy:         args[1],
				personalization: args[2],
				ad1:             args[3],
				ad2:             args[4],
				nonce:           args[5],
			}.process()
		},
	}
}

func cmdCtrDrbgReseedAft() command {
	return command{
		requiredArgs: 8, // Output length, entropy, personalization, reseedAD, reseedEntropy, ad1, ad2, nonce
		handler: func(args [][]byte) ([][]byte, error) {
			return acvpCtrDrbg{
				outLen:          binary.LittleEndian.Uint32(args[0]),
				entropy:         args[1],
				personalization: args[2],
				reseedAd:        args[3],
				reseedEntropy:   args[4],
				ad1:             args[5],
				ad2:             args[6],
				nonce:           args[7],
			}.process()
		},
	}
}

type acvpCtrDrbg struct {
	outLen          uint32
	entropy         []byte
	personalization []byte
	ad1             []byte
	ad2             []byte
	nonce           []byte
	reseedAd        []byte // May be empty for no reseed
	reseedEntropy   []byte // May be empty for no reseed
}

func (args acvpCtrDrbg) process() ([][]byte, error) {
	// Our capability describes no personalization support.
	if len(args.personalization) > 0 {
		return nil, errors.New("personalization string not supported")
	}

	// Our capability describes no derivation function support, so the nonce
	// should be empty.
	if len(args.nonce) > 0 {
		return nil, errors.New("unexpected nonce value")
	}

	// Our capability describes entropy input len of 384 bits.
	entropy, err := require48Bytes(args.entropy)
	if err != nil {
		return nil, fmt.Errorf("entropy: %w", err)
	}

	// Our capability describes additional input len of 384 bits.
	ad1, err := require48Bytes(args.ad1)
	if err != nil {
		return nil, fmt.Errorf("AD1: %w", err)
	}
	ad2, err := require48Bytes(args.ad2)
	if err != nil {
		return nil, fmt.Errorf("AD2: %w", err)
	}

	withReseed := len(args.reseedAd) > 0
	var reseedAd, reseedEntropy *[48]byte
	if withReseed {
		// Ditto RE: entropy and additional data lengths for reseeding.
		if reseedAd, err = require48Bytes(args.reseedAd); err != nil {
			return nil, fmt.Errorf("reseed AD: %w", err)
		}
		if reseedEntropy, err = require48Bytes(args.reseedEntropy); err != nil {
			return nil, fmt.Errorf("reseed entropy: %w", err)
		}
	}

	// Our capabilities describe no prediction resistance and allow both
	// reseed and no reseed, so the test procedure is:
	//   * Instantiate DRBG
	//   * Reseed (if enabled)
	//   * Generate but don't output
	//   * Generate output
	//   * Uninstantiate
	// See Table 7 in draft-vassilev-acvp-drbg
	out := make([]byte, args.outLen)
	ctrDrbg := drbg.NewCounter(entropy)
	if withReseed {
		ctrDrbg.Reseed(reseedEntropy, reseedAd)
	}
	ctrDrbg.Generate(out, ad1)
	ctrDrbg.Generate(out, ad2)

	return [][]byte{out}, nil
}

// Verify input is 48 byte slice, and cast it to a pointer to a fixed-size array
// of 48 bytes, or return an error.
func require48Bytes(input []byte) (*[48]byte, error) {
	if inputLen := len(input); inputLen != 48 {
		return nil, fmt.Errorf("invalid length: %d", inputLen)
	}
	return (*[48]byte)(input), nil
}

func cmdKdfCounterAft() command {
	return command{
		requiredArgs: 5, // Number output bytes, PRF name, counter location string, key, number of counter bits
		handler: func(args [][]byte) ([][]byte, error) {
			outputBytes := binary.LittleEndian.Uint32(args[0])
			prf := args[1]
			counterLocation := args[2]
			key := args[3]
			counterBits := binary.LittleEndian.Uint32(args[4])

			if outputBytes != 32 {
				return nil, fmt.Errorf("KDF received unsupported output length %d bytes", outputBytes)
			}
			if !bytes.Equal(prf, []byte("CMAC-AES128")) && !bytes.Equal(prf, []byte("CMAC-AES192")) && !bytes.Equal(prf, []byte("CMAC-AES256")) {
				return nil, fmt.Errorf("KDF received unsupported PRF %q", string(prf))
			}
			if !bytes.Equal(counterLocation, []byte("before fixed data")) {
				return nil, fmt.Errorf("KDF received unsupported counter location %q", string(counterLocation))
			}
			// The spec doesn't describe the "deferred" property for a KDF counterMode test case.
			// BoringSSL's acvptool sends an empty key when deferred=true, but with the capabilities
			// we register all test cases ahve deferred=false and provide a key from the populated
			// keyIn property.
			if len(key) == 0 {
				return nil, errors.New("deferred test cases are not supported")
			}
			if counterBits != 16 {
				return nil, fmt.Errorf("KDF received unsupported counter length %d", counterBits)
			}

			block, err := aes.New(key)
			if err != nil {
				return nil, fmt.Errorf("failed to create cipher: %v", err)
			}
			kdf := gcm.NewCounterKDF(block)

			var label byte
			var context [12]byte
			rand.Reader.Read(context[:])

			result := kdf.DeriveKey(label, context)

			fixedData := make([]byte, 1+1+12) // 1 byte label, 1 null byte, 12 bytes context.
			fixedData[0] = label
			copy(fixedData[2:], context[:])

			return [][]byte{key, fixedData, result[:]}, nil
		},
	}
}

func cmdKdfFeedbackAft() command {
	return command{
		requiredArgs: 5, // Number output bytes, PRF name, counter location string, key, number of counter bits, IV
		handler: func(args [][]byte) ([][]byte, error) {
			// The max supported output len for the KDF algorithm type is 4096 bits, making an int cast
			// here safe.
			// See https://pages.nist.gov/ACVP/draft-celi-acvp-kbkdf.html#section-7.3.2
			outputBytes := int(binary.LittleEndian.Uint32(args[0]))
			prf := string(args[1])
			counterLocation := args[2]
			key := args[3]
			counterBits := binary.LittleEndian.Uint32(args[4])

			if !strings.HasPrefix(prf, "HMAC-") {
				return nil, fmt.Errorf("feedback KDF received unsupported PRF %q", prf)
			}
			prf = prf[len("HMAC-"):]

			h, err := lookupHash(prf)
			if err != nil {
				return nil, fmt.Errorf("feedback KDF received unsupported PRF %q: %w", prf, err)
			}

			if !bytes.Equal(counterLocation, []byte("after fixed data")) {
				return nil, fmt.Errorf("feedback KDF received unsupported counter location %q", string(counterLocation))
			}

			// The spec doesn't describe the "deferred" property for a KDF counterMode test case.
			// BoringSSL's acvptool sends an empty key when deferred=true, but with the capabilities
			// we register all test cases have deferred=false and provide a key from the populated
			// keyIn property.
			if len(key) == 0 {
				return nil, errors.New("deferred test cases are not supported")
			}

			if counterBits != 8 {
				return nil, fmt.Errorf("feedback KDF received unsupported counter length %d", counterBits)
			}

			var context [12]byte
			rand.Reader.Read(context[:])
			fixedData := make([]byte, 1+1+12) // 1 byte label (we pick null), 1 null byte, 12 bytes context.
			copy(fixedData[2:], context[:])

			result := hkdf.Expand(h, key, string(fixedData[:]), outputBytes)

			return [][]byte{key, fixedData[:], result[:]}, nil
		},
	}
}

func cmdRsaKeyGenAft() command {
	return command{
		requiredArgs: 1, // Modulus bit-size
		handler: func(args [][]byte) ([][]byte, error) {
			bitSize := binary.LittleEndian.Uint32(args[0])

			key, err := getRSAKey((int)(bitSize))
			if err != nil {
				return nil, fmt.Errorf("generating RSA key: %w", err)
			}

			N, e, d, P, Q, _, _, _ := key.Export()

			eBytes := make([]byte, 4)
			binary.BigEndian.PutUint32(eBytes, uint32(e))

			return [][]byte{eBytes, P, Q, N, d}, nil
		},
	}
}

func cmdRsaSigGenAft(hashFunc func() fips140.Hash, hashName string, pss bool) command {
	return command{
		requiredArgs: 2, // Modulus bit-size, message
		handler: func(args [][]byte) ([][]byte, error) {
			bitSize := binary.LittleEndian.Uint32(args[0])
			msg := args[1]

			key, err := getRSAKey((int)(bitSize))
			if err != nil {
				return nil, fmt.Errorf("generating RSA key: %w", err)
			}

			h := hashFunc()
			h.Write(msg)
			digest := h.Sum(nil)

			var sig []byte
			if !pss {
				sig, err = rsa.SignPKCS1v15(key, hashName, digest)
				if err != nil {
					return nil, fmt.Errorf("signing RSA message: %w", err)
				}
			} else {
				sig, err = rsa.SignPSS(rand.Reader, key, hashFunc(), digest, h.Size())
				if err != nil {
					return nil, fmt.Errorf("signing RSA message: %w", err)
				}
			}

			N, e, _, _, _, _, _, _ := key.Export()
			eBytes := make([]byte, 4)
			binary.BigEndian.PutUint32(eBytes, uint32(e))

			return [][]byte{N, eBytes, sig}, nil
		},
	}
}

func cmdRsaSigVerAft(hashFunc func() fips140.Hash, hashName string, pss bool) command {
	return command{
		requiredArgs: 4, // n, e, message, signature
		handler: func(args [][]byte) ([][]byte, error) {
			nBytes := args[0]
			eBytes := args[1]
			msg := args[2]
			sig := args[3]

			paddedE := make([]byte, 4)
			copy(paddedE[4-len(eBytes):], eBytes)
			e := int(binary.BigEndian.Uint32(paddedE))

			n, err := bigmod.NewModulus(nBytes)
			if err != nil {
				return nil, fmt.Errorf("invalid RSA modulus: %w", err)
			}

			pub := &rsa.PublicKey{
				N: n,
				E: e,
			}

			h := hashFunc()
			h.Write(msg)
			digest := h.Sum(nil)

			if !pss {
				err = rsa.VerifyPKCS1v15(pub, hashName, digest, sig)
			} else {
				err = rsa.VerifyPSS(pub, hashFunc(), digest, sig)
			}
			if err != nil {
				return [][]byte{{0}}, nil
			}

			return [][]byte{{1}}, nil
		},
	}
}

// rsaKeyCache caches generated keys by modulus bit-size.
var rsaKeyCache = map[int]*rsa.PrivateKey{}

// getRSAKey returns a cached RSA private key with the specified modulus bit-size
// or generates one if necessary.
func getRSAKey(bits int) (*rsa.PrivateKey, error) {
	if key, exists := rsaKeyCache[bits]; exists {
		return key, nil
	}

	key, err := rsa.GenerateKey(rand.Reader, bits)
	if err != nil {
		return nil, err
	}

	rsaKeyCache[bits] = key
	return key, nil
}

func cmdOneStepNoCounterHmacAft(h func() fips140.Hash) command {
	return command{
		requiredArgs: 4, // key, info, salt, outBytes
		handler: func(args [][]byte) ([][]byte, error) {
			key := args[0]
			info := args[1]
			salt := args[2]
			outBytes := binary.LittleEndian.Uint32(args[3])

			mac := hmac.New(h, salt)
			mac.Size()

			if outBytes != uint32(mac.Size()) {
				return nil, fmt.Errorf("invalid output length: got %d, want %d", outBytes, mac.Size())
			}

			data := make([]byte, 0, len(key)+len(info))
			data = append(data, key...)
			data = append(data, info...)

			mac.Write(data)
			out := mac.Sum(nil)

			return [][]byte{out}, nil
		},
	}
}

func cmdKtsIfcInitiatorAft(h func() fips140.Hash) command {
	return command{
		requiredArgs: 3, // output bytes, n bytes, e bytes
		handler: func(args [][]byte) ([][]byte, error) {
			outputBytes := binary.LittleEndian.Uint32(args[0])
			nBytes := args[1]
			eBytes := args[2]

			n, err := bigmod.NewModulus(nBytes)
			if err != nil {
				return nil, fmt.Errorf("invalid RSA modulus: %w", err)
			}

			paddedE := make([]byte, 4)
			copy(paddedE[4-len(eBytes):], eBytes)
			e := int(binary.BigEndian.Uint32(paddedE))
			if e != 0x10001 {
				return nil, errors.New("e must be 0x10001")
			}

			pub := &rsa.PublicKey{
				N: n,
				E: e,
			}

			dkm := make([]byte, outputBytes)
			if _, err := rand.Read(dkm); err != nil {
				return nil, fmt.Errorf("failed to generate random DKM: %v", err)
			}

			iutC, err := rsa.EncryptOAEP(h(), h(), rand.Reader, pub, dkm, nil)
			if err != nil {
				return nil, fmt.Errorf("OAEP encryption failed: %v", err)
			}

			return [][]byte{iutC, dkm}, nil
		},
	}
}

func cmdKtsIfcResponderAft(h func() fips140.Hash) command {
	return command{
		requiredArgs: 6, // n bytes, e bytes, p bytes, q bytes, d bytes, c bytes
		handler: func(args [][]byte) ([][]byte, error) {
			nBytes := args[0]
			eBytes := args[1]

			pBytes := args[2]
			qBytes := args[3]
			dBytes := args[4]

			cBytes := args[5]

			paddedE := make([]byte, 4)
			copy(paddedE[4-len(eBytes):], eBytes)
			e := int(binary.BigEndian.Uint32(paddedE))
			if e != 0x10001 {
				return nil, errors.New("e must be 0x10001")
			}

			priv, err := rsa.NewPrivateKey(nBytes, int(e), dBytes, pBytes, qBytes)
			if err != nil {
				return nil, fmt.Errorf("failed to create private key: %v", err)
			}

			dkm, err := rsa.DecryptOAEP(h(), h(), priv, cBytes, nil)
			if err != nil {
				return nil, fmt.Errorf("OAEP decryption failed: %v", err)
			}

			return [][]byte{dkm}, nil
		},
	}
}

func TestACVP(t *testing.T) {
	testenv.SkipIfShortAndSlow(t)

	const (
		bsslModule    = "boringssl.googlesource.com/boringssl.git"
		bsslVersion   = "v0.0.0-20250207174145-0bb19f6126cb"
		goAcvpModule  = "github.com/cpu/go-acvp"
		goAcvpVersion = "v0.0.0-20250126154732-de1ba727a0be"
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
	cmd.Env = append(os.Environ(),
		"ACVP_WRAPPER=1",
		"GODEBUG=fips140=on",
	)
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
