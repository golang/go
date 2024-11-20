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
		return 3, 80, true
	}
	return 0, 0, false // A mismatch
}

func hashToBytes[P Point[P]](c *Curve[P], dst, hash []byte) {
	e := bigmod.NewNat()
	hashToNat(c, e, hash)
	copy(dst, e.Bytes(c.N))
}

func sign[P Point[P]](c *Curve[P], priv *PrivateKey, drbg *hmacDRBG, hash []byte) (*Signature, error) {
	functionCode, blockSize, ok := canUseKDSA(c.curve)
	if !ok {
		return signGeneric(c, priv, drbg, hash)
	}
	for {
		k, _, err := randomPoint(c, func(b []byte) error {
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
		// the parameter block.
		hashToBytes(c, params[2*blockSize:3*blockSize], hash)
		copy(params[3*blockSize+blockSize-len(priv.d):], priv.d)
		copy(params[4*blockSize:5*blockSize], k.Bytes(c.N))
		// Convert verify function code into a sign function code by adding 8.
		// We also need to set the 'deterministic' bit in the function code, by
		// adding 128, in order to stop the instruction using its own random number
		// generator in addition to the random number we supply.
		switch kdsa(functionCode+136, &params) {
		case 0: // success
			return &Signature{R: params[:blockSize], S: params[blockSize : 2*blockSize]}, nil
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
	copy(params[0*blockSize+blockSize-len(r):], r)
	copy(params[1*blockSize+blockSize-len(s):], s)
	hashToBytes(c, params[2*blockSize:3*blockSize], hash)
	copy(params[3*blockSize:5*blockSize], pub.q[1:]) // strip 0x04 prefix
	if kdsa(functionCode, &params) != 0 {
		return errors.New("invalid signature")
	}
	return nil
}
