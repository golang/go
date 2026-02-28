// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build boringcrypto

// Note: Can run these tests against the non-BoringCrypto
// version of the code by using "CGO_ENABLED=0 go test".

package rsa

import (
	"crypto"
	"crypto/rand"
	"encoding/asn1"
	"encoding/hex"
	"math/big"
	"runtime"
	"runtime/debug"
	"sync"
	"testing"
)

func TestBoringASN1Marshal(t *testing.T) {
	t.Setenv("GODEBUG", "rsa1024min=0")

	k, err := GenerateKey(rand.Reader, 128)
	if err != nil {
		t.Fatal(err)
	}
	_, err = asn1.Marshal(k.PublicKey)
	if err != nil {
		t.Fatal(err)
	}
}

func TestBoringVerify(t *testing.T) {
	// Check that signatures that lack leading zeroes don't verify.
	key := &PublicKey{
		N: bigFromHex("c4fdf7b40a5477f206e6ee278eaef888ca73bf9128a9eef9f2f1ddb8b7b71a4c07cfa241f028a04edb405e4d916c61d6beabc333813dc7b484d2b3c52ee233c6a79b1eea4e9cc51596ba9cd5ac5aeb9df62d86ea051055b79d03f8a4fa9f38386f5bd17529138f3325d46801514ea9047977e0829ed728e68636802796801be1"),
		E: 65537,
	}

	hash := fromHex("019c5571724fb5d0e47a4260c940e9803ba05a44")
	paddedHash := fromHex("3021300906052b0e03021a05000414019c5571724fb5d0e47a4260c940e9803ba05a44")

	// signature is one byte shorter than key.N.
	sig := fromHex("5edfbeb6a73e7225ad3cc52724e2872e04260d7daf0d693c170d8c4b243b8767bc7785763533febc62ec2600c30603c433c095453ede59ff2fcabeb84ce32e0ed9d5cf15ffcbc816202b64370d4d77c1e9077d74e94a16fb4fa2e5bec23a56d7a73cf275f91691ae1801a976fcde09e981a2f6327ac27ea1fecf3185df0d56")

	err := VerifyPKCS1v15(key, 0, paddedHash, sig)
	if err == nil {
		t.Errorf("raw: expected verification error")
	}

	err = VerifyPKCS1v15(key, crypto.SHA1, hash, sig)
	if err == nil {
		t.Errorf("sha1: expected verification error")
	}
}

func BenchmarkBoringVerify(b *testing.B) {
	// Check that signatures that lack leading zeroes don't verify.
	key := &PublicKey{
		N: bigFromHex("c4fdf7b40a5477f206e6ee278eaef888ca73bf9128a9eef9f2f1ddb8b7b71a4c07cfa241f028a04edb405e4d916c61d6beabc333813dc7b484d2b3c52ee233c6a79b1eea4e9cc51596ba9cd5ac5aeb9df62d86ea051055b79d03f8a4fa9f38386f5bd17529138f3325d46801514ea9047977e0829ed728e68636802796801be1"),
		E: 65537,
	}

	hash := fromHex("019c5571724fb5d0e47a4260c940e9803ba05a44")

	// signature is one byte shorter than key.N.
	sig := fromHex("5edfbeb6a73e7225ad3cc52724e2872e04260d7daf0d693c170d8c4b243b8767bc7785763533febc62ec2600c30603c433c095453ede59ff2fcabeb84ce32e0ed9d5cf15ffcbc816202b64370d4d77c1e9077d74e94a16fb4fa2e5bec23a56d7a73cf275f91691ae1801a976fcde09e981a2f6327ac27ea1fecf3185df0d56")

	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		err := VerifyPKCS1v15(key, crypto.SHA1, hash, sig)
		if err == nil {
			b.Fatalf("sha1: expected verification error")
		}
	}
}

func TestBoringGenerateKey(t *testing.T) {
	k, err := GenerateKey(rand.Reader, 2048) // 2048 is smallest size BoringCrypto might kick in for
	if err != nil {
		t.Fatal(err)
	}

	// Non-Boring GenerateKey always sets CRTValues to a non-nil (possibly empty) slice.
	if k.Precomputed.CRTValues == nil {
		t.Fatalf("GenerateKey: Precomputed.CRTValues = nil")
	}
}

func TestBoringFinalizers(t *testing.T) {
	if runtime.GOOS == "nacl" || runtime.GOOS == "js" {
		// Times out on nacl and js/wasm (without BoringCrypto)
		// but not clear why - probably consuming rand.Reader too quickly
		// and being throttled. Also doesn't really matter.
		t.Skipf("skipping on %s/%s", runtime.GOOS, runtime.GOARCH)
	}

	k, err := GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}

	// Run test with GOGC=10, to make bug more likely.
	// Without the KeepAlives, the loop usually dies after
	// about 30 iterations.
	defer debug.SetGCPercent(debug.SetGCPercent(10))
	for n := 0; n < 200; n++ {
		// Clear the underlying BoringCrypto object cache.
		privCache.Clear()

		// Race to create the underlying BoringCrypto object.
		// The ones that lose the race are prime candidates for
		// being GC'ed too early if the finalizers are not being
		// used correctly.
		var wg sync.WaitGroup
		for i := 0; i < 10; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				sum := make([]byte, 32)
				_, err := SignPKCS1v15(rand.Reader, k, crypto.SHA256, sum)
				if err != nil {
					panic(err) // usually caused by memory corruption, so hard stop
				}
			}()
		}
		wg.Wait()
	}
}

func bigFromHex(hex string) *big.Int {
	n, ok := new(big.Int).SetString(hex, 16)
	if !ok {
		panic("bad hex: " + hex)
	}
	return n
}

func fromHex(hexStr string) []byte {
	s, err := hex.DecodeString(hexStr)
	if err != nil {
		panic(err)
	}
	return s
}
