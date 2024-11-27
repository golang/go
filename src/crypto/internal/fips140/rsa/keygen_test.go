// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rsa

import (
	"bufio"
	"encoding/hex"
	"fmt"
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

func decodeHex(t *testing.T, s string) []byte {
	t.Helper()
	b, err := hex.DecodeString(s)
	if err != nil {
		t.Fatalf("failed to decode hex %q: %v", s, err)
	}
	return b
}
