// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ecdh_test

import (
	"bytes"
	"crypto"
	"crypto/cipher"
	"crypto/ecdh"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"internal/testenv"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"golang.org/x/crypto/chacha20"
)

// Check that PublicKey and PrivateKey implement the interfaces documented in
// crypto.PublicKey and crypto.PrivateKey.
var _ interface {
	Equal(x crypto.PublicKey) bool
} = &ecdh.PublicKey{}
var _ interface {
	Public() crypto.PublicKey
	Equal(x crypto.PrivateKey) bool
} = &ecdh.PrivateKey{}

func TestECDH(t *testing.T) {
	testAllCurves(t, func(t *testing.T, curve ecdh.Curve) {
		aliceKey, err := curve.GenerateKey(rand.Reader)
		if err != nil {
			t.Fatal(err)
		}
		bobKey, err := curve.GenerateKey(rand.Reader)
		if err != nil {
			t.Fatal(err)
		}

		alicePubKey, err := curve.NewPublicKey(aliceKey.PublicKey().Bytes())
		if err != nil {
			t.Error(err)
		}
		if !bytes.Equal(aliceKey.PublicKey().Bytes(), alicePubKey.Bytes()) {
			t.Error("encoded and decoded public keys are different")
		}
		if !aliceKey.PublicKey().Equal(alicePubKey) {
			t.Error("encoded and decoded public keys are different")
		}

		alicePrivKey, err := curve.NewPrivateKey(aliceKey.Bytes())
		if err != nil {
			t.Error(err)
		}
		if !bytes.Equal(aliceKey.Bytes(), alicePrivKey.Bytes()) {
			t.Error("encoded and decoded private keys are different")
		}
		if !aliceKey.Equal(alicePrivKey) {
			t.Error("encoded and decoded private keys are different")
		}

		bobSecret, err := curve.ECDH(bobKey, aliceKey.PublicKey())
		if err != nil {
			t.Fatal(err)
		}
		aliceSecret, err := curve.ECDH(aliceKey, bobKey.PublicKey())
		if err != nil {
			t.Fatal(err)
		}

		if !bytes.Equal(bobSecret, aliceSecret) {
			t.Error("two ECDH computations came out different")
		}
	})
}

type countingReader struct {
	r io.Reader
	n int
}

func (r *countingReader) Read(p []byte) (int, error) {
	n, err := r.r.Read(p)
	r.n += n
	return n, err
}

func TestGenerateKey(t *testing.T) {
	testAllCurves(t, func(t *testing.T, curve ecdh.Curve) {
		r := &countingReader{r: rand.Reader}
		k, err := curve.GenerateKey(r)
		if err != nil {
			t.Fatal(err)
		}

		// GenerateKey does rejection sampling. If the masking works correctly,
		// the probability of a rejection is 1-ord(G)/2^ceil(log2(ord(G))),
		// which for all curves is small enough (at most 2^-32, for P-256) that
		// a bit flip is more likely to make this test fail than bad luck.
		// Account for the extra MaybeReadByte byte, too.
		if got, expected := r.n, len(k.Bytes())+1; got > expected {
			t.Errorf("expected GenerateKey to consume at most %v bytes, got %v", expected, got)
		}
	})
}

var vectors = map[ecdh.Curve]struct {
	PrivateKey, PublicKey string
	PeerPublicKey         string
	SharedSecret          string
}{
	// NIST vectors from CAVS 14.1, ECC CDH Primitive (SP800-56A).
	ecdh.P256(): {
		PrivateKey: "7d7dc5f71eb29ddaf80d6214632eeae03d9058af1fb6d22ed80badb62bc1a534",
		PublicKey: "04ead218590119e8876b29146ff89ca61770c4edbbf97d38ce385ed281d8a6b230" +
			"28af61281fd35e2fa7002523acc85a429cb06ee6648325389f59edfce1405141",
		PeerPublicKey: "04700c48f77f56584c5cc632ca65640db91b6bacce3a4df6b42ce7cc838833d287" +
			"db71e509e3fd9b060ddb20ba5c51dcc5948d46fbf640dfe0441782cab85fa4ac",
		SharedSecret: "46fc62106420ff012e54a434fbdd2d25ccc5852060561e68040dd7778997bd7b",
	},
	ecdh.P384(): {
		PrivateKey: "3cc3122a68f0d95027ad38c067916ba0eb8c38894d22e1b15618b6818a661774ad463b205da88cf699ab4d43c9cf98a1",
		PublicKey: "049803807f2f6d2fd966cdd0290bd410c0190352fbec7ff6247de1302df86f25d34fe4a97bef60cff548355c015dbb3e5f" +
			"ba26ca69ec2f5b5d9dad20cc9da711383a9dbe34ea3fa5a2af75b46502629ad54dd8b7d73a8abb06a3a3be47d650cc99",
		PeerPublicKey: "04a7c76b970c3b5fe8b05d2838ae04ab47697b9eaf52e764592efda27fe7513272734466b400091adbf2d68c58e0c50066" +
			"ac68f19f2e1cb879aed43a9969b91a0839c4c38a49749b661efedf243451915ed0905a32b060992b468c64766fc8437a",
		SharedSecret: "5f9d29dc5e31a163060356213669c8ce132e22f57c9a04f40ba7fcead493b457e5621e766c40a2e3d4d6a04b25e533f1",
	},
	// For some reason all field elements in the test vector (both scalars and
	// base field elements), but not the shared secret output, have two extra
	// leading zero bytes (which in big-endian are irrelevant). Removed here.
	ecdh.P521(): {
		PrivateKey: "017eecc07ab4b329068fba65e56a1f8890aa935e57134ae0ffcce802735151f4eac6564f6ee9974c5e6887a1fefee5743ae2241bfeb95d5ce31ddcb6f9edb4d6fc47",
		PublicKey: "0400602f9d0cf9e526b29e22381c203c48a886c2b0673033366314f1ffbcba240ba42f4ef38a76174635f91e6b4ed34275eb01c8467d05ca80315bf1a7bbd945f550a5" +
			"01b7c85f26f5d4b2d7355cf6b02117659943762b6d1db5ab4f1dbc44ce7b2946eb6c7de342962893fd387d1b73d7a8672d1f236961170b7eb3579953ee5cdc88cd2d",
		PeerPublicKey: "0400685a48e86c79f0f0875f7bc18d25eb5fc8c0b07e5da4f4370f3a9490340854334b1e1b87fa395464c60626124a4e70d0f785601d37c09870ebf176666877a2046d" +
			"01ba52c56fc8776d9e8f5db4f0cc27636d0b741bbe05400697942e80b739884a83bde99e0f6716939e632bc8986fa18dccd443a348b6c3e522497955a4f3c302f676",
		SharedSecret: "005fc70477c3e63bc3954bd0df3ea0d1f41ee21746ed95fc5e1fdf90930d5e136672d72cc770742d1711c3c3a4c334a0ad9759436a4d3c5bf6e74b9578fac148c831",
	},
	// X25519 test vector from RFC 7748, Section 6.1.
	ecdh.X25519(): {
		PrivateKey:    "77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a",
		PublicKey:     "8520f0098930a754748b7ddcb43ef75a0dbf3a0d26381af4eba4a98eaa9b4e6a",
		PeerPublicKey: "de9edb7d7b7dc1b4d35b61c2ece435373f8343c85b78674dadfc7e146f882b4f",
		SharedSecret:  "4a5d9d5ba4ce2de1728e3bf480350f25e07e21c947d19e3376f09b3c1e161742",
	},
}

func TestVectors(t *testing.T) {
	testAllCurves(t, func(t *testing.T, curve ecdh.Curve) {
		v := vectors[curve]
		key, err := curve.NewPrivateKey(hexDecode(t, v.PrivateKey))
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(key.PublicKey().Bytes(), hexDecode(t, v.PublicKey)) {
			t.Error("public key derived from the private key does not match")
		}
		peer, err := curve.NewPublicKey(hexDecode(t, v.PeerPublicKey))
		if err != nil {
			t.Fatal(err)
		}
		secret, err := curve.ECDH(key, peer)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(secret, hexDecode(t, v.SharedSecret)) {
			t.Error("shared secret does not match")
		}
	})
}

func hexDecode(t *testing.T, s string) []byte {
	b, err := hex.DecodeString(s)
	if err != nil {
		t.Fatal("invalid hex string:", s)
	}
	return b
}

func TestString(t *testing.T) {
	testAllCurves(t, func(t *testing.T, curve ecdh.Curve) {
		s := fmt.Sprintf("%s", curve)
		if s[:1] != "P" && s[:1] != "X" {
			t.Errorf("unexpected Curve string encoding: %q", s)
		}
	})
}

func testAllCurves(t *testing.T, f func(t *testing.T, curve ecdh.Curve)) {
	t.Run("P256", func(t *testing.T) { f(t, ecdh.P256()) })
	t.Run("P384", func(t *testing.T) { f(t, ecdh.P384()) })
	t.Run("P521", func(t *testing.T) { f(t, ecdh.P521()) })
	t.Run("X25519", func(t *testing.T) { f(t, ecdh.X25519()) })
}

func BenchmarkECDH(b *testing.B) {
	benchmarkAllCurves(b, func(b *testing.B, curve ecdh.Curve) {
		c, err := chacha20.NewUnauthenticatedCipher(make([]byte, 32), make([]byte, 12))
		if err != nil {
			b.Fatal(err)
		}
		rand := cipher.StreamReader{
			S: c, R: zeroReader,
		}

		peerKey, err := curve.GenerateKey(rand)
		if err != nil {
			b.Fatal(err)
		}
		peerShare := peerKey.PublicKey().Bytes()
		b.ResetTimer()
		b.ReportAllocs()

		var allocationsSink byte

		for i := 0; i < b.N; i++ {
			key, err := curve.GenerateKey(rand)
			if err != nil {
				b.Fatal(err)
			}
			share := key.PublicKey().Bytes()
			peerPubKey, err := curve.NewPublicKey(peerShare)
			if err != nil {
				b.Fatal(err)
			}
			secret, err := curve.ECDH(key, peerPubKey)
			if err != nil {
				b.Fatal(err)
			}
			allocationsSink ^= secret[0] ^ share[0]
		}
	})
}

func benchmarkAllCurves(b *testing.B, f func(b *testing.B, curve ecdh.Curve)) {
	b.Run("P256", func(b *testing.B) { f(b, ecdh.P256()) })
	b.Run("P384", func(b *testing.B) { f(b, ecdh.P384()) })
	b.Run("P521", func(b *testing.B) { f(b, ecdh.P521()) })
	b.Run("X25519", func(b *testing.B) { f(b, ecdh.X25519()) })
}

type zr struct{}

// Read replaces the contents of dst with zeros. It is safe for concurrent use.
func (zr) Read(dst []byte) (n int, err error) {
	for i := range dst {
		dst[i] = 0
	}
	return len(dst), nil
}

var zeroReader = zr{}

const linkerTestProgram = `
package main
import "crypto/ecdh"
import "crypto/rand"
func main() {
	curve := ecdh.P384()
	key, err := curve.GenerateKey(rand.Reader)
	if err != nil { panic(err) }
	_, err = curve.NewPublicKey(key.PublicKey().Bytes())
	if err != nil { panic(err) }
	_, err = curve.NewPrivateKey(key.Bytes())
	if err != nil { panic(err) }
	_, err = curve.ECDH(key, key.PublicKey())
	if err != nil { panic(err) }
	println("OK")
}
`

// TestLinker ensures that using one curve does not bring all other
// implementations into the binary. This also guarantees that govulncheck can
// avoid warning about a curve-specific vulnerability if that curve is not used.
func TestLinker(t *testing.T) {
	if testing.Short() {
		t.Skip("test requires running 'go build'")
	}
	testenv.MustHaveGoBuild(t)

	dir := t.TempDir()
	hello := filepath.Join(dir, "hello.go")
	err := os.WriteFile(hello, []byte(linkerTestProgram), 0664)
	if err != nil {
		t.Fatal(err)
	}

	run := func(args ...string) string {
		cmd := exec.Command(args[0], args[1:]...)
		cmd.Dir = dir
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("%v: %v\n%s", args, err, string(out))
		}
		return string(out)
	}

	goBin := testenv.GoToolPath(t)
	run(goBin, "build", "-o", "hello.exe", "hello.go")
	if out := run("./hello.exe"); out != "OK\n" {
		t.Error("unexpected output:", out)
	}

	// List all text symbols under crypto/... and make sure there are some for
	// P384, but none for the other curves.
	var consistent bool
	nm := run(goBin, "tool", "nm", "hello.exe")
	for _, match := range regexp.MustCompile(`(?m)T (crypto/.*)$`).FindAllStringSubmatch(nm, -1) {
		symbol := strings.ToLower(match[1])
		if strings.Contains(symbol, "p384") {
			consistent = true
		}
		if strings.Contains(symbol, "p224") || strings.Contains(symbol, "p256") || strings.Contains(symbol, "p521") {
			t.Errorf("unexpected symbol in program using only ecdh.P384: %s", match[1])
		}
	}
	if !consistent {
		t.Error("no P384 symbols found in program using ecdh.P384, test is broken")
	}
}
