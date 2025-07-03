// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rsa

import (
	"bufio"
	"crypto/internal/fips140/bigmod"
	"encoding/hex"
	"fmt"
	"math/big"
	"os"
	"strings"
	"testing"
)

func TestMillerRabin(t *testing.T) {
	f, err := os.Open("testdata/miller_rabin_tests.txt")
	if err != nil {
		t.Fatal(err)
	}

	var expected bool
	var W, B string
	var lineNum int
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		lineNum++
		line := scanner.Text()
		if len(line) == 0 || line[0] == '#' {
			continue
		}

		k, v, _ := strings.Cut(line, " = ")
		switch k {
		case "Result":
			switch v {
			case "Composite":
				expected = millerRabinCOMPOSITE
			case "PossiblyPrime":
				expected = millerRabinPOSSIBLYPRIME
			default:
				t.Fatalf("unknown result %q on line %d", v, lineNum)
			}
		case "W":
			W = v
		case "B":
			B = v

			t.Run(fmt.Sprintf("line %d", lineNum), func(t *testing.T) {
				if len(W)%2 != 0 {
					W = "0" + W
				}
				for len(B) < len(W) {
					B = "0" + B
				}

				mr, err := millerRabinSetup(decodeHex(t, W))
				if err != nil {
					t.Logf("W = %s", W)
					t.Logf("B = %s", B)
					t.Fatalf("failed to set up Miller-Rabin test: %v", err)
				}

				result, err := millerRabinIteration(mr, decodeHex(t, B))
				if err != nil {
					t.Logf("W = %s", W)
					t.Logf("B = %s", B)
					t.Fatalf("failed to run Miller-Rabin test: %v", err)
				}

				if result != expected {
					t.Logf("W = %s", W)
					t.Logf("B = %s", B)
					t.Fatalf("unexpected result: got %v, want %v", result, expected)
				}
			})
		default:
			t.Fatalf("unknown key %q on line %d", k, lineNum)
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
}

func TestTotient(t *testing.T) {
	f, err := os.Open("testdata/gcd_lcm_tests.txt")
	if err != nil {
		t.Fatal(err)
	}

	var GCD, A, B, LCM string
	var lineNum int
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		lineNum++
		line := scanner.Text()
		if len(line) == 0 || line[0] == '#' {
			continue
		}

		k, v, _ := strings.Cut(line, " = ")
		switch k {
		case "GCD":
			GCD = v
		case "A":
			A = v
		case "B":
			B = v
		case "LCM":
			LCM = v

			t.Run(fmt.Sprintf("line %d", lineNum), func(t *testing.T) {
				if A == "0" || B == "0" {
					t.Skip("skipping test with zero input")
				}
				if LCM == "1" {
					t.Skip("skipping test with LCM=1")
				}

				p, _ := bigmod.NewModulus(addOne(decodeHex(t, A)))
				a, _ := bigmod.NewNat().SetBytes(decodeHex(t, A), p)
				q, _ := bigmod.NewModulus(addOne(decodeHex(t, B)))
				b, _ := bigmod.NewNat().SetBytes(decodeHex(t, B), q)

				gcd, err := bigmod.NewNat().GCDVarTime(a, b)
				// GCD doesn't work if a and b are both even, but LCM handles it.
				if err == nil {
					if got := strings.TrimLeft(hex.EncodeToString(gcd.Bytes(p)), "0"); got != GCD {
						t.Fatalf("unexpected GCD: got %s, want %s", got, GCD)
					}
				}

				lcm, err := totient(p, q)
				if oddDivisorLargerThan32Bits(decodeHex(t, GCD)) {
					if err != errDivisorTooLarge {
						t.Fatalf("expected divisor too large error, got %v", err)
					}
					t.Skip("GCD too large")
				}
				if err != nil {
					t.Fatalf("failed to calculate totient: %v", err)
				}
				if got := strings.TrimLeft(hex.EncodeToString(lcm.Nat().Bytes(lcm)), "0"); got != LCM {
					t.Fatalf("unexpected LCM: got %s, want %s", got, LCM)
				}
			})
		default:
			t.Fatalf("unknown key %q on line %d", k, lineNum)
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
}

func oddDivisorLargerThan32Bits(b []byte) bool {
	x := new(big.Int).SetBytes(b)
	x.Rsh(x, x.TrailingZeroBits())
	return x.BitLen() > 32
}

func addOne(b []byte) []byte {
	x := new(big.Int).SetBytes(b)
	x.Add(x, big.NewInt(1))
	return x.Bytes()
}

func decodeHex(t *testing.T, s string) []byte {
	t.Helper()
	if len(s)%2 != 0 {
		s = "0" + s
	}
	b, err := hex.DecodeString(s)
	if err != nil {
		t.Fatalf("failed to decode hex %q: %v", s, err)
	}
	return b
}
