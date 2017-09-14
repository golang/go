// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Note: Can run these tests against the non-BoringCrypto
// version of the code by using "CGO_ENABLED=0 go test".

package rsa

import (
	"bytes"
	"crypto"
	"crypto/rand"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/asn1"
	"encoding/hex"
	"reflect"
	"sync"
	"testing"
	"unsafe"
)

func TestBoringASN1Marshal(t *testing.T) {
	k, err := GenerateKey(rand.Reader, 128)
	if err != nil {
		t.Fatal(err)
	}
	// This used to fail, because of the unexported 'boring' field.
	// Now the compiler hides it [sic].
	_, err = asn1.Marshal(k.PublicKey)
	if err != nil {
		t.Fatal(err)
	}
}

func TestBoringDeepEqual(t *testing.T) {
	k, err := GenerateKey(rand.Reader, 128)
	if err != nil {
		t.Fatal(err)
	}
	k.boring = nil // probably nil already but just in case
	k2 := *k
	k2.boring = unsafe.Pointer(k) // anything not nil, for this test
	if !reflect.DeepEqual(k, &k2) {
		// compiler should be hiding the boring field from reflection
		t.Fatalf("DeepEqual compared boring fields")
	}
}

func TestBoringVerify(t *testing.T) {
	// This changed behavior and broke golang.org/x/crypto/openpgp.
	// Go accepts signatures without leading 0 padding, while BoringCrypto does not.
	// So the Go wrappers must adapt.
	key := &PublicKey{
		N: bigFromHex("c4fdf7b40a5477f206e6ee278eaef888ca73bf9128a9eef9f2f1ddb8b7b71a4c07cfa241f028a04edb405e4d916c61d6beabc333813dc7b484d2b3c52ee233c6a79b1eea4e9cc51596ba9cd5ac5aeb9df62d86ea051055b79d03f8a4fa9f38386f5bd17529138f3325d46801514ea9047977e0829ed728e68636802796801be1"),
		E: 65537,
	}

	hash := fromHex("019c5571724fb5d0e47a4260c940e9803ba05a44")
	paddedHash := fromHex("3021300906052b0e03021a05000414019c5571724fb5d0e47a4260c940e9803ba05a44")

	// signature is one byte shorter than key.N.
	sig := fromHex("5edfbeb6a73e7225ad3cc52724e2872e04260d7daf0d693c170d8c4b243b8767bc7785763533febc62ec2600c30603c433c095453ede59ff2fcabeb84ce32e0ed9d5cf15ffcbc816202b64370d4d77c1e9077d74e94a16fb4fa2e5bec23a56d7a73cf275f91691ae1801a976fcde09e981a2f6327ac27ea1fecf3185df0d56")

	err := VerifyPKCS1v15(key, 0, paddedHash, sig)
	if err != nil {
		t.Errorf("raw: %v", err)
	}

	err = VerifyPKCS1v15(key, crypto.SHA1, hash, sig)
	if err != nil {
		t.Errorf("sha1: %v", err)
	}
}

// The goal for BoringCrypto is to be indistinguishable from standard Go crypto.
// Test that when routines are passed a not-actually-random reader, they
// consume and potentially expose the expected bits from that reader.
// This is awful but it makes sure that golden tests based on deterministic
// "randomness" sources are unchanged by BoringCrypto.
//
// For decryption and signing, r is only used for blinding,
// so we can and do still use BoringCrypto with its own true
// randomness source, but we must be careful to consume
// from r as if we'd used it for blinding.

type testRandReader struct {
	t      *testing.T
	offset int64
	seq    [8]byte
	data   []byte
	buf    [32]byte
}

func (r *testRandReader) Read(b []byte) (int, error) {
	if len(r.data) == 0 && len(b) > 0 {
		for i := range r.seq {
			r.seq[i]++
			if r.seq[i] != 0 {
				break
			}
		}
		r.buf = sha256.Sum256(r.seq[:])
		r.data = r.buf[:]
	}
	n := copy(b, r.data)
	r.data = r.data[n:]
	r.offset += int64(n)
	return n, nil
}

func (r *testRandReader) checkOffset(offset int64) {
	r.t.Helper()
	if r.offset != offset {
		r.t.Fatalf("r.offset = %d, expected %d", r.offset, offset)
	}
}

func testRand(t *testing.T) *testRandReader {
	return &testRandReader{t: t}
}

var testKeyCache struct {
	once sync.Once
	k    *PrivateKey
}

func testKey(t *testing.T) *PrivateKey {
	testKeyCache.once.Do(func() {
		// Note: Key must be 2048 bits in order to trigger
		// BoringCrypto code paths.
		k, err := GenerateKey(testRand(t), 2048)
		if err != nil {
			t.Fatal(err)
		}
		testKeyCache.k = k
	})
	return testKeyCache.k
}

func bytesFromHex(t *testing.T, x string) []byte {
	b, err := hex.DecodeString(x)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestBoringRandGenerateKey(t *testing.T) {
	r := testRand(t)
	k, err := GenerateKey(r, 2048) // 2048 is smallest size BoringCrypto might kick in for
	if err != nil {
		t.Fatal(err)
	}
	n := bigFromHex("b2e9c4c8b1c0f03ba6994fe1e715a3e598f0571f4676da420615b7b997d431ea7535ceb98e6b52172fe0d2fccfc5f696d1b34144f7d19d85633fcbf56daff805a66457b360b1b0f40ec18fb83f4c9b86f1b5fe26b209cdfff26911a95047df797210969693226423915c9be53ff1c06f86fe2d228273ef25970b90a3c70979f9d68458d5dd38f6700436f7cd5939c04be3e1f2ff52272513171540a685c9e8c8e20694e529cc3e0cc13d2fb91ac499d44b920a03e42be89a15e7ca73c29f2e2a1a8a7d9be57516ccb95e878db6ce6096e386a793cccc19eba15a37cc0f1234b7a25ee7c87569bc74c7ef3d6ad8d84a5ddb1e8901ae593f945523fe5e0ed451a5")
	if k.N.Cmp(n) != 0 {
		t.Fatalf("GenerateKey: wrong N\nhave %x\nwant %x", k.N, n)
	}
	r.checkOffset(35200)

	// Non-Boring GenerateKey always sets CRTValues to a non-nil (possibly empty) slice.
	if k.Precomputed.CRTValues == nil {
		t.Fatalf("GenerateKey: Precomputed.CRTValues = nil")
	}
}

func TestBoringRandGenerateMultiPrimeKey(t *testing.T) {
	r := testRand(t)
	k, err := GenerateMultiPrimeKey(r, 2, 2048)
	if err != nil {
		t.Fatal(err)
	}
	n := bigFromHex("b2e9c4c8b1c0f03ba6994fe1e715a3e598f0571f4676da420615b7b997d431ea7535ceb98e6b52172fe0d2fccfc5f696d1b34144f7d19d85633fcbf56daff805a66457b360b1b0f40ec18fb83f4c9b86f1b5fe26b209cdfff26911a95047df797210969693226423915c9be53ff1c06f86fe2d228273ef25970b90a3c70979f9d68458d5dd38f6700436f7cd5939c04be3e1f2ff52272513171540a685c9e8c8e20694e529cc3e0cc13d2fb91ac499d44b920a03e42be89a15e7ca73c29f2e2a1a8a7d9be57516ccb95e878db6ce6096e386a793cccc19eba15a37cc0f1234b7a25ee7c87569bc74c7ef3d6ad8d84a5ddb1e8901ae593f945523fe5e0ed451a5")
	if k.N.Cmp(n) != 0 {
		t.Fatalf("GenerateKey: wrong N\nhave %x\nwant %x", k.N, n)
	}
	r.checkOffset(35200)
}

func TestBoringRandEncryptPKCS1v15(t *testing.T) {
	r := testRand(t)
	k := testKey(t)
	enc, err := EncryptPKCS1v15(r, &k.PublicKey, []byte("hello world"))
	if err != nil {
		t.Fatal(err)
	}
	want := bytesFromHex(t, "a8c8c0d248e669942a140c1184e1112afbf794b7427d9ac966bd2dbb4c05a2fee76f311f7feec743b8a8715e34bf741b0d0c4226559daf4de258ff712178e3f25fecb7d3eee90251e8ae4b4b7b907cd2763948cc9da34ce83c69934b523830545a536c1ba4d3740f4687e877acee9c768bcd8e88d472ba5d905493121f4830d95dcea36ef0f1223ffb0a9008eddfc53aca36877328924a2c631dce4b67e745564301fe51ab2c768b39e525bda1e1a08e029b58c53a0b92285f734592d2deebda957bcfd29c697aee263fce5c5023c7d3495b6a9114a8ac691aa661721cf45973b68678bb1e15d6605b9040951163d5b6df0d7f0b20dcefa251a7a8947a090f4b")
	if !bytes.Equal(enc, want) {
		t.Fatalf("EncryptPKCS1v15: wrong enc\nhave %x\nwant %x", enc, want)
	}
	r.checkOffset(242)
}

func TestBoringRandDecryptPKCS1v15(t *testing.T) {
	r := testRand(t)
	k := testKey(t)
	enc := bytesFromHex(t, "a8c8c0d248e669942a140c1184e1112afbf794b7427d9ac966bd2dbb4c05a2fee76f311f7feec743b8a8715e34bf741b0d0c4226559daf4de258ff712178e3f25fecb7d3eee90251e8ae4b4b7b907cd2763948cc9da34ce83c69934b523830545a536c1ba4d3740f4687e877acee9c768bcd8e88d472ba5d905493121f4830d95dcea36ef0f1223ffb0a9008eddfc53aca36877328924a2c631dce4b67e745564301fe51ab2c768b39e525bda1e1a08e029b58c53a0b92285f734592d2deebda957bcfd29c697aee263fce5c5023c7d3495b6a9114a8ac691aa661721cf45973b68678bb1e15d6605b9040951163d5b6df0d7f0b20dcefa251a7a8947a090f4b")
	dec, err := DecryptPKCS1v15(r, k, enc)
	if err != nil {
		t.Fatal(err)
	}
	want := []byte("hello world")
	if !bytes.Equal(dec, want) {
		t.Fatalf("DecryptPKCS1v15: wrong dec\nhave %x\nwant %x", dec, want)
	}
	r.checkOffset(256)
}

func TestBoringRandDecryptPKCS1v15SessionKey(t *testing.T) {
	r := testRand(t)
	k := testKey(t)
	enc := bytesFromHex(t, "a8c8c0d248e669942a140c1184e1112afbf794b7427d9ac966bd2dbb4c05a2fee76f311f7feec743b8a8715e34bf741b0d0c4226559daf4de258ff712178e3f25fecb7d3eee90251e8ae4b4b7b907cd2763948cc9da34ce83c69934b523830545a536c1ba4d3740f4687e877acee9c768bcd8e88d472ba5d905493121f4830d95dcea36ef0f1223ffb0a9008eddfc53aca36877328924a2c631dce4b67e745564301fe51ab2c768b39e525bda1e1a08e029b58c53a0b92285f734592d2deebda957bcfd29c697aee263fce5c5023c7d3495b6a9114a8ac691aa661721cf45973b68678bb1e15d6605b9040951163d5b6df0d7f0b20dcefa251a7a8947a090f4b")
	dec := make([]byte, 11)
	err := DecryptPKCS1v15SessionKey(r, k, enc, dec)
	if err != nil {
		t.Fatal(err)
	}
	want := []byte("hello world")
	if !bytes.Equal(dec, want) {
		t.Fatalf("DecryptPKCS1v15SessionKey: wrong dec\nhave %x\nwant %x", dec, want)
	}
	r.checkOffset(256)
}

func TestBoringRandSignPKCS1v15(t *testing.T) {
	r := testRand(t)
	k := testKey(t)
	sum := sha1.Sum([]byte("hello"))
	sig, err := SignPKCS1v15(r, k, crypto.SHA1, sum[:])
	if err != nil {
		t.Fatal(err)
	}
	want := bytesFromHex(t, "4a8da3c0c41af2b8a93d011d4e11f4da9b2d52641c6c3d78d863987e857295adcedfae0e0d3ec00352bd134dc3fbb93b23a1fbe3718775762d78165bbbd37c6ef8e07bfa44e16ed2f1b05ebc04ba7bd60162d8689edb8709349e06bc281d34c2a3ee75d3454bfd95053cbb27c10515fb9132290a6ecc858e0c003201a9e100aac7f66af967364a1176e4ed9ef672d41481c59580f98bb82f205f712153fd5e3035a811da9d6e56e50609d1d604857f6d8e958bb84f354cfa28e0b8bcbb1261f929382d431454f07cbf60c18ff1243b11c6b552f3a0aa7e936f45cded40688ee53b1b630f944139f4f51baae49cd039b57b2b82f58f5589335137f4b09bd315f5")
	if !bytes.Equal(sig, want) {
		t.Fatalf("SignPKCS1v15(hash=SHA1): wrong sig\nhave %x\nwant %x", sig, want)
	}

	sig, err = SignPKCS1v15(r, k, 0, sum[:])
	if err != nil {
		t.Fatal(err)
	}
	want = bytesFromHex(t, "5d3d34495ffade926adab2de0545aaf1f22a03def949b69e1c91d34a2f0c7f2d682af46034151a1b67aa22cb9c1a8cc24c1358fce9ac6a2141879bbe107371b14faa97b12494260d9602ed1355f22ab3495b0bb7c137bc6801c1113fc2bdc00d4c250bbd8fa17e4ff86f71544b30a78e9d62c0b949afd1159760282c2700ec8be24cd884efd585ec55b45506d90e66cc3c5911baaea961e6c4e8018c4b4feb04afdd71880e3d8eff120288e53289a1bfb9fe7a3b9aca1d4549f133063647bfd4c6f4c0f4038f1bbcb4d112aa601f1b15402595076adfdbefb1bb64d3193bafb0305145bb536cd949a03ebe0470c6a155369f784afab2e25e9d5c03d8e13dcf1a")
	if !bytes.Equal(sig, want) {
		t.Fatalf("SignPKCS1v15(hash=0): wrong sig\nhave %x\nwant %x", sig, want)
	}
	r.checkOffset(768)
}

func TestBoringRandSignPSS(t *testing.T) {
	r := testRand(t)
	k := testKey(t)
	sum := sha1.Sum([]byte("hello"))
	sig, err := SignPSS(r, k, crypto.SHA1, sum[:], nil)
	if err != nil {
		t.Fatal(err)
	}
	want := bytesFromHex(t, "a0de84c9654c2e78e33c899090f8dc0590046fda4ee29d133340800596401ae0df61bf8aa5689df3f873ad13cf55df5209c3a8c6450918b74c2017f87c2d588809740622c7752e3153a26d04bd3e9d9f6daa676e8e5e65a8a11d4fbd271d4693ab6a303652328dc1c923b484fa179fd6d9e8b523da74f3a307531c0dd75f243a041f7df22414dfdb83b3a241fe73e7af0f95cb6b60831bdd46dc05618e5cb3653476eb7d5405fa5ca98dad8f787ca86179055f305daa87eb424671878a93965e47d3002e2774be311d696b42e5691eddb2f788cd35246b408eb5d045c891ba1d57ce4c6fc935ceec90f7999406252f6266957cce4e7f12cf0ec94af358aeefa7")
	if !bytes.Equal(sig, want) {
		t.Fatalf("SignPSS: wrong sig\nhave %x\nwant %x", sig, want)
	}
	r.checkOffset(490)
}

func TestBoringRandEncryptOAEP(t *testing.T) {
	r := testRand(t)
	k := testKey(t)
	enc, err := EncryptOAEP(sha256.New(), r, &k.PublicKey, []byte("hello"), []byte("label"))
	if err != nil {
		t.Fatal(err)
	}
	want := bytesFromHex(t, "55dc7b590a511c2d249232ecbb70040e8e0ec03206caae5ec0a401a0ad8013209ef546870f93d0946b9845ace092d456d092403f76f12ee65c2b8759731a25589d8a7e857407d09cfbe36ae36fc4daeb514ac597b1de2f7dc8450ab78a9e420c9b5dbbae3e402c8f378bd35505a47d556b705ab8985707a22e3583c172ef5730f05fd0845880d67c1ddd3c1525aa4c2c4e162bd6435a485609f6bd76c8ff73a7b5d043e4724458594703245fabdb479ef2786c757b35932a645399f2703647785b59b971970e6bccef3e6cd6fae39f9f135203eb104f0db20cf48e461cb7d824889c0d5d6a47cd0bf213c2f7acb3ddbd3effefebb4f60458ffc8b6ff1e4cc447")
	if !bytes.Equal(enc, want) {
		t.Fatalf("EncryptOAEP: wrong enc\nhave %x\nwant %x", enc, want)
	}
	r.checkOffset(32)
}

func TestBoringRandDecryptOAEP(t *testing.T) {
	r := testRand(t)
	k := testKey(t)
	enc := bytesFromHex(t, "55dc7b590a511c2d249232ecbb70040e8e0ec03206caae5ec0a401a0ad8013209ef546870f93d0946b9845ace092d456d092403f76f12ee65c2b8759731a25589d8a7e857407d09cfbe36ae36fc4daeb514ac597b1de2f7dc8450ab78a9e420c9b5dbbae3e402c8f378bd35505a47d556b705ab8985707a22e3583c172ef5730f05fd0845880d67c1ddd3c1525aa4c2c4e162bd6435a485609f6bd76c8ff73a7b5d043e4724458594703245fabdb479ef2786c757b35932a645399f2703647785b59b971970e6bccef3e6cd6fae39f9f135203eb104f0db20cf48e461cb7d824889c0d5d6a47cd0bf213c2f7acb3ddbd3effefebb4f60458ffc8b6ff1e4cc447")
	dec, err := DecryptOAEP(sha256.New(), r, k, enc, []byte("label"))
	if err != nil {
		t.Fatal(err)
	}
	want := []byte("hello")
	if !bytes.Equal(dec, want) {
		t.Fatalf("DecryptOAEP: wrong dec\nhave %x\nwant %x", dec, want)
	}
	r.checkOffset(256)
}
