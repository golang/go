// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

package ecdsa

import (
	"crypto/internal/fips140/bigmod"
	"crypto/internal/fips140deps/cpu"
	"crypto/internal/impl"
	"errors"
)

// kdsa invokes the "compute digital signature authentication"
// instruction with the given function code and 4096 byte
// parameter block.
//
// The return value corresponds to the condition code set by the
// instruction. Interrupted invocations are handled by the
// function.
//
//go:noescape
func kdsa(fc uint64, params *[4096]byte) (errn uint64)

var supportsKDSA = cpu.S390XHasECDSA

func init() {
	// CP Assist for Cryptographic Functions (CPACF)
	// https://www.ibm.com/docs/en/zos/3.1.0?topic=icsf-cp-assist-cryptographic-functions-cpacf
	impl.Register("ecdsa", "CPACF", &supportsKDSA)
}

// canUseKDSA checks if KDSA instruction is available, and if it is, it checks
// the name of the curve to see if it matches the curves supported(P-256, P-384, P-521).
// Then, based on the curve name, a function code and a block size will be assigned.
// If KDSA instruction is not available or if the curve is not supported, canUseKDSA
// will set ok to false.
func canUseKDSA(c curveID) (functionCode uint64, blockSize int, ok bool) {
	if !supportsKDSA {
		return 0, 0, false
	}
	switch c {
	case p256:
		return 1, 32, true
	case p384:
		return 2, 48, true
	case p521:
		// Note that the block size doesn't match the field size for P-521.
		return 3, 80, true
	}
	return 0, 0, false // A mismatch
}

func hashToBytes[P Point[P]](c *Curve[P], hash []byte) []byte {
	e := bigmod.NewNat()
	hashToNat(c, e, hash)
	return e.Bytes(c.N)
}

// randomScalar is a copy of [randomPoint] that doesn't call ScalarBaseMult.
func randomScalar[P Point[P]](c *Curve[P], generate func([]byte) error) (k *bigmod.Nat, err error) {
	for {
		b := make([]byte, c.N.Size())
		if err := generate(b); err != nil {
			return nil, err
		}
		if excess := len(b)*8 - c.N.BitLen(); excess > 0 {
			if c.curve != p521 {
				panic("ecdsa: internal error: unexpectedly masking off bits")
			}
			b = rightShift(b, excess)
		}
		if k, err := bigmod.NewNat().SetBytes(b, c.N); err == nil && k.IsZero() == 0 {
			return k, nil
		}
	}
}

func appendBlock(p []byte, blocksize int, b []byte) []byte {
	if len(b) > blocksize {
		panic("ecdsa: internal error: appendBlock input larger than block")
	}
	padding := blocksize - len(b)
	p = append(p, make([]byte, padding)...)
	return append(p, b...)
}

func trimBlock(p []byte, size int) ([]byte, error) {
	for _, b := range p[:len(p)-size] {
		if b != 0 {
			return nil, errors.New("ecdsa: internal error: KDSA produced invalid signature")
		}
	}
	return p[len(p)-size:], nil
}

func sign[P Point[P]](c *Curve[P], priv *PrivateKey, drbg *hmacDRBG, hash []byte) (*Signature, error) {
	functionCode, blockSize, ok := canUseKDSA(c.curve)
	if !ok {
		return signGeneric(c, priv, drbg, hash)
	}
	for {
		k, err := randomScalar(c, func(b []byte) error {
			drbg.Generate(b)
			return nil
		})
		if err != nil {
			return nil, err
		}

		// The parameter block looks like the following for sign.
		// 	+---------------------+
		// 	|   Signature(R)      |
		//	+---------------------+
		//	|   Signature(S)      |
		//	+---------------------+
		//	|   Hashed Message    |
		//	+---------------------+
		//	|   Private Key       |
		//	+---------------------+
		//	|   Random Number     |
		//	+---------------------+
		//	|                     |
		//	|        ...          |
		//	|                     |
		//	+---------------------+
		// The common components(signatureR, signatureS, hashedMessage, privateKey and
		// random number) each takes block size of bytes. The block size is different for
		// different curves and is set by canUseKDSA function.
		var params [4096]byte

		// Copy content into the parameter block. In the sign case,
		// we copy hashed message, private key and random number into
		// the parameter block. We skip the signature slots.
		p := params[:2*blockSize]
		p = appendBlock(p, blockSize, hashToBytes(c, hash))
		p = appendBlock(p, blockSize, priv.d)
		p = appendBlock(p, blockSize, k.Bytes(c.N))
		// Convert verify function code into a sign function code by adding 8.
		// We also need to set the 'deterministic' bit in the function code, by
		// adding 128, in order to stop the instruction using its own random number
		// generator in addition to the random number we supply.
		switch kdsa(functionCode+136, &params) {
		case 0: // success
			elementSize := (c.N.BitLen() + 7) / 8
			r, err := trimBlock(params[:blockSize], elementSize)
			if err != nil {
				return nil, err
			}
			s, err := trimBlock(params[blockSize:2*blockSize], elementSize)
			if err != nil {
				return nil, err
			}
			return &Signature{R: r, S: s}, nil
		case 1: // error
			return nil, errors.New("zero parameter")
		case 2: // retry
			continue
		}
	}
}

func verify[P Point[P]](c *Curve[P], pub *PublicKey, hash []byte, sig *Signature) error {
	functionCode, blockSize, ok := canUseKDSA(c.curve)
	if !ok {
		return verifyGeneric(c, pub, hash, sig)
	}

	r, s := sig.R, sig.S
	if len(r) > blockSize || len(s) > blockSize {
		return errors.New("invalid signature")
	}

	// The parameter block looks like the following for verify:
	// 	+---------------------+
	// 	|   Signature(R)      |
	//	+---------------------+
	//	|   Signature(S)      |
	//	+---------------------+
	//	|   Hashed Message    |
	//	+---------------------+
	//	|   Public Key X      |
	//	+---------------------+
	//	|   Public Key Y      |
	//	+---------------------+
	//	|                     |
	//	|        ...          |
	//	|                     |
	//	+---------------------+
	// The common components(signatureR, signatureS, hashed message, public key X,
	// and public key Y) each takes block size of bytes. The block size is different for
	// different curves and is set by canUseKDSA function.
	var params [4096]byte

	// Copy content into the parameter block. In the verify case,
	// we copy signature (r), signature(s), hashed message, public key x component,
	// and public key y component into the parameter block.
	p := params[:0]
	p = appendBlock(p, blockSize, r)
	p = appendBlock(p, blockSize, s)
	p = appendBlock(p, blockSize, hashToBytes(c, hash))
	p = appendBlock(p, blockSize, pub.q[1:1+len(pub.q)/2])
	p = appendBlock(p, blockSize, pub.q[1+len(pub.q)/2:])
	if kdsa(functionCode, &params) != 0 {
		return errors.New("invalid signature")
	}
	return nil
}
