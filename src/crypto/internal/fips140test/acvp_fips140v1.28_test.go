// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !fips140v1.0 && !fips140v1.26

package fipstest

import (
	"crypto/internal/fips140/bigmod"
	"crypto/internal/fips140/rsa"
	"crypto/internal/fips140/sha256"
	"crypto/internal/fips140/sha3"
	"crypto/internal/fips140/sha512"
	"crypto/rand"

	_ "embed"
	"encoding/binary"
	"errors"
	"fmt"
	"hash"
	"math/big"
)

//go:embed acvp_capabilities_fips140v1.28.json
var capabilitiesJson []byte

var testConfigFile = "acvp_test_fips140v1.28.config.json"

func init() {
	// RSA keyGen for keyFormat=crt capability. Previous versions used standard
	commands["RSA/keyGen/crt"] = cmdRsaKeyGenCrtAft()

	// RSA sigGen additional hash algorithms (adds SHA2-512 truncated hashes, SHA-3)
	// Support for PSS saltLen argument.
	commands["RSA/sigGen/SHA2-224/pkcs1v1.5"] = cmdRsaSigGenSaltLenAft(func() hash.Hash { return sha256.New224() }, "SHA-224", false)
	commands["RSA/sigGen/SHA2-256/pkcs1v1.5"] = cmdRsaSigGenSaltLenAft(func() hash.Hash { return sha256.New() }, "SHA-256", false)
	commands["RSA/sigGen/SHA2-384/pkcs1v1.5"] = cmdRsaSigGenSaltLenAft(func() hash.Hash { return sha512.New384() }, "SHA-384", false)
	commands["RSA/sigGen/SHA2-512/pkcs1v1.5"] = cmdRsaSigGenSaltLenAft(func() hash.Hash { return sha512.New() }, "SHA-512", false)
	commands["RSA/sigGen/SHA2-512/224/pkcs1v1.5"] = cmdRsaSigGenSaltLenAft(func() hash.Hash { return sha512.New512_224() }, "SHA-512/224", false)
	commands["RSA/sigGen/SHA2-512/256/pkcs1v1.5"] = cmdRsaSigGenSaltLenAft(func() hash.Hash { return sha512.New512_256() }, "SHA-512/256", false)
	commands["RSA/sigGen/SHA3-224/pkcs1v1.5"] = cmdRsaSigGenSaltLenAft(func() hash.Hash { return sha3.New224() }, "SHA3-224", false)
	commands["RSA/sigGen/SHA3-256/pkcs1v1.5"] = cmdRsaSigGenSaltLenAft(func() hash.Hash { return sha3.New256() }, "SHA3-256", false)
	commands["RSA/sigGen/SHA3-384/pkcs1v1.5"] = cmdRsaSigGenSaltLenAft(func() hash.Hash { return sha3.New384() }, "SHA3-384", false)
	commands["RSA/sigGen/SHA3-512/pkcs1v1.5"] = cmdRsaSigGenSaltLenAft(func() hash.Hash { return sha3.New512() }, "SHA3-512", false)
	commands["RSA/sigGen/SHA2-512/224/pss"] = cmdRsaSigGenSaltLenAft(func() hash.Hash { return sha512.New512_224() }, "SHA-512/224", true)
	commands["RSA/sigGen/SHA2-512/256/pss"] = cmdRsaSigGenSaltLenAft(func() hash.Hash { return sha512.New512_256() }, "SHA-512/256", true)
	commands["RSA/sigGen/SHA3-224/pss"] = cmdRsaSigGenSaltLenAft(func() hash.Hash { return sha3.New224() }, "SHA3-224", true)
	commands["RSA/sigGen/SHA3-256/pss"] = cmdRsaSigGenSaltLenAft(func() hash.Hash { return sha3.New256() }, "SHA3-256", true)
	commands["RSA/sigGen/SHA3-384/pss"] = cmdRsaSigGenSaltLenAft(func() hash.Hash { return sha3.New384() }, "SHA3-384", true)
	commands["RSA/sigGen/SHA3-512/pss"] = cmdRsaSigGenSaltLenAft(func() hash.Hash { return sha3.New512() }, "SHA3-512", true)
	commands["RSA/sigGen/SHA2-224/pss"] = cmdRsaSigGenSaltLenAft(func() hash.Hash { return sha256.New224() }, "SHA-224", true)
	commands["RSA/sigGen/SHA2-256/pss"] = cmdRsaSigGenSaltLenAft(func() hash.Hash { return sha256.New() }, "SHA-256", true)
	commands["RSA/sigGen/SHA2-384/pss"] = cmdRsaSigGenSaltLenAft(func() hash.Hash { return sha512.New384() }, "SHA-384", true)
	commands["RSA/sigGen/SHA2-512/pss"] = cmdRsaSigGenSaltLenAft(func() hash.Hash { return sha512.New() }, "SHA-512", true)

	// RSA sigGen with pubExpMode=random capability. Previous versions used fixed.
	// Also adds SHA2-512 truncated hashes. Notably SHA-3 is not supported by ACVP
	// for RSA sigVer.
	commands["RSA/sigVer/SHA2-224/pkcs1v1.5"] = cmdRsaSigVerRandExpAft(func() hash.Hash { return sha256.New224() }, "SHA-224", false)
	commands["RSA/sigVer/SHA2-256/pkcs1v1.5"] = cmdRsaSigVerRandExpAft(func() hash.Hash { return sha256.New() }, "SHA-256", false)
	commands["RSA/sigVer/SHA2-384/pkcs1v1.5"] = cmdRsaSigVerRandExpAft(func() hash.Hash { return sha512.New384() }, "SHA-384", false)
	commands["RSA/sigVer/SHA2-512/pkcs1v1.5"] = cmdRsaSigVerRandExpAft(func() hash.Hash { return sha512.New() }, "SHA-512", false)
	commands["RSA/sigVer/SHA2-512/224/pkcs1v1.5"] = cmdRsaSigVerRandExpAft(func() hash.Hash { return sha512.New512_224() }, "SHA-512/224", false)
	commands["RSA/sigVer/SHA2-512/256/pkcs1v1.5"] = cmdRsaSigVerRandExpAft(func() hash.Hash { return sha512.New512_256() }, "SHA-512/256", false)
	commands["RSA/sigVer/SHA2-224/pss"] = cmdRsaSigVerRandExpAft(func() hash.Hash { return sha256.New224() }, "SHA-224", true)
	commands["RSA/sigVer/SHA2-256/pss"] = cmdRsaSigVerRandExpAft(func() hash.Hash { return sha256.New() }, "SHA-256", true)
	commands["RSA/sigVer/SHA2-384/pss"] = cmdRsaSigVerRandExpAft(func() hash.Hash { return sha512.New384() }, "SHA-384", true)
	commands["RSA/sigVer/SHA2-512/pss"] = cmdRsaSigVerRandExpAft(func() hash.Hash { return sha512.New() }, "SHA-512", true)
	commands["RSA/sigVer/SHA2-512/224/pss"] = cmdRsaSigVerRandExpAft(func() hash.Hash { return sha512.New512_224() }, "SHA-512/224", true)
	commands["RSA/sigVer/SHA2-512/256/pss"] = cmdRsaSigVerRandExpAft(func() hash.Hash { return sha512.New512_256() }, "SHA-512/256", true)

	// KTS-IFC with keyGenerationMethods=rsakpg2-crt (random exponent, CRT format).
	// Previous version used rsakpg1-basic (fixed exponent, basic format).
	commands["KTS-IFC/SHA2-224/responder/crt"] = cmdKtsIfcResponderRandExpCrtAft(func() hash.Hash { return sha256.New224() })
	commands["KTS-IFC/SHA2-256/responder/crt"] = cmdKtsIfcResponderRandExpCrtAft(func() hash.Hash { return sha256.New() })
	commands["KTS-IFC/SHA2-384/responder/crt"] = cmdKtsIfcResponderRandExpCrtAft(func() hash.Hash { return sha512.New384() })
	commands["KTS-IFC/SHA2-512/responder/crt"] = cmdKtsIfcResponderRandExpCrtAft(func() hash.Hash { return sha512.New() })
	commands["KTS-IFC/SHA2-512/224/responder/crt"] = cmdKtsIfcResponderRandExpCrtAft(func() hash.Hash { return sha512.New512_224() })
	commands["KTS-IFC/SHA2-512/256/responder/crt"] = cmdKtsIfcResponderRandExpCrtAft(func() hash.Hash { return sha512.New512_256() })
	commands["KTS-IFC/SHA3-224/responder/crt"] = cmdKtsIfcResponderRandExpCrtAft(func() hash.Hash { return sha3.New224() })
	commands["KTS-IFC/SHA3-256/responder/crt"] = cmdKtsIfcResponderRandExpCrtAft(func() hash.Hash { return sha3.New256() })
	commands["KTS-IFC/SHA3-384/responder/crt"] = cmdKtsIfcResponderRandExpCrtAft(func() hash.Hash { return sha3.New384() })
	commands["KTS-IFC/SHA3-512/responder/crt"] = cmdKtsIfcResponderRandExpCrtAft(func() hash.Hash { return sha3.New512() })
}

func cmdRsaKeyGenCrtAft() command {
	return command{
		requiredArgs: 1, // Modulus bit-size
		handler: func(args [][]byte) ([][]byte, error) {
			bitSize := binary.LittleEndian.Uint32(args[0])

			key, err := getRSAKey((int)(bitSize))
			if err != nil {
				return nil, fmt.Errorf("generating RSA key: %w", err)
			}

			N, e, d, P, Q, dP, dQ, qInv := key.Export()

			eBytes := make([]byte, 4)
			binary.BigEndian.PutUint32(eBytes, uint32(e))

			return [][]byte{eBytes, P, Q, N, d, dP, dQ, qInv}, nil
		},
	}
}

func cmdRsaSigGenSaltLenAft(hashFunc func() hash.Hash, hashName string, pss bool) command {
	return command{
		requiredArgs: 3, // Modulus bit-size, message, saltLen
		handler: func(args [][]byte) ([][]byte, error) {
			bitSize := binary.LittleEndian.Uint32(args[0])
			msg := args[1]
			saltLen := binary.LittleEndian.Uint32(args[2])

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
				sig, err = rsa.SignPSS(rand.Reader, key, hashFunc(), digest, (int)(saltLen))
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

func cmdRsaSigVerRandExpAft(hashFunc func() hash.Hash, hashName string, pss bool) command {
	return command{
		requiredArgs: 4, // n, e, message, signature
		handler: func(args [][]byte) ([][]byte, error) {
			nBytes := args[0]
			eBytes := args[1]
			msg := args[2]
			sig := args[3]

			n, err := bigmod.NewModulus(nBytes)
			if err != nil {
				return nil, fmt.Errorf("invalid RSA modulus: %w", err)
			}

			h := hashFunc()
			h.Write(msg)
			digest := h.Sum(nil)

			if len(eBytes) < 32/8 || (len(eBytes) == 32/8 && eBytes[0] < 0x80) {
				paddedE := make([]byte, 4)
				copy(paddedE[4-len(eBytes):], eBytes)
				e := int(binary.BigEndian.Uint32(paddedE))

				pub := &rsa.PublicKey{
					N: n,
					E: e,
				}

				if !pss {
					err = rsa.VerifyPKCS1v15(pub, hashName, digest, sig)
				} else {
					err = rsa.VerifyPSS(pub, hashFunc(), digest, sig)
				}
			} else {
				pub := &rsa.TestingOnlyLargeExponentPublicKey{
					N: n,
					E: eBytes,
				}

				if !pss {
					err = rsa.TestingOnlyLargeExponentVerifyPKCS1v15(pub, hashName, digest, sig)
				} else {
					err = rsa.TestingOnlyLargeExponentVerifyPSS(pub, hashFunc(), digest, sig)
				}
			}

			if err != nil {
				return [][]byte{{0}}, nil
			}

			return [][]byte{{1}}, nil
		},
	}
}

func cmdKtsIfcResponderRandExpCrtAft(h func() hash.Hash) command {
	return command{
		requiredArgs: 8, // n bytes, e bytes, p bytes, q bytes, dmp1 bytes, dmq1 bytes, iqmp bytes, c bytes
		handler: func(args [][]byte) ([][]byte, error) {
			nBytes := args[0]
			eBytes := args[1]

			pBytes := args[2]
			qBytes := args[3]

			dmp1Bytes := args[4]
			dmq1Bytes := args[5]
			iqmpBytes := args[6]

			cBytes := args[7]

			// We need to compute 'd' from the CRT values for NewPrivateKeyWithPrecomputation to
			// check consistency.
			p, q := new(big.Int).SetBytes(pBytes), new(big.Int).SetBytes(qBytes)
			dP, dQ := new(big.Int).SetBytes(dmp1Bytes), new(big.Int).SetBytes(dmq1Bytes)
			pMinus1, qMinus1 := new(big.Int).Sub(p, big.NewInt(1)), new(big.Int).Sub(q, big.NewInt(1))
			x, y := new(big.Int), new(big.Int)
			gcd := new(big.Int).GCD(x, y, pMinus1, qMinus1)
			diff := new(big.Int).Sub(dQ, dP)
			if new(big.Int).Mod(diff, gcd).Sign() != 0 {
				return nil, errors.New("inconsistent CRT parameters")
			}
			lcm := new(big.Int).Mul(pMinus1, qMinus1)
			lcm.Div(lcm, gcd)
			d := new(big.Int).Mul(x, pMinus1)
			d.Mul(d, diff).Div(d, gcd)
			d.Add(d, dP).Mod(d, lcm)

			var dkm []byte
			var err error

			if len(eBytes) < 32/8 || (len(eBytes) == 32/8 && eBytes[0] < 0x80) {
				// Exponent fits in an int32, use standard PrivateKey.
				paddedE := make([]byte, 4)
				copy(paddedE[4-len(eBytes):], eBytes)
				e := int(binary.BigEndian.Uint32(paddedE))

				priv, err := rsa.NewPrivateKeyWithPrecomputation(nBytes, e, d.Bytes(), pBytes, qBytes, dmp1Bytes, dmq1Bytes, iqmpBytes)
				if err != nil {
					return nil, fmt.Errorf("failed to create private key: %v", err)
				}

				dkm, err = rsa.DecryptOAEP(h(), h(), priv, cBytes, nil)
			} else {
				// Large exponent, use TestingOnlyLargeExponentPrivateKey.
				priv, err := rsa.TestingOnlyNewLargeExponentPrivateKeyWithPrecomputation(nBytes, eBytes, d.Bytes(), pBytes, qBytes, dmp1Bytes, dmq1Bytes, iqmpBytes)
				if err != nil {
					return nil, fmt.Errorf("failed to create private key: %v", err)
				}

				dkm, err = rsa.TestingOnlyLargeExponentDecryptOAEP(h(), h(), priv, cBytes, nil)
			}
			if err != nil {
				return nil, fmt.Errorf("OAEP decryption failed: %v", err)
			}

			return [][]byte{dkm}, nil
		},
	}
}
