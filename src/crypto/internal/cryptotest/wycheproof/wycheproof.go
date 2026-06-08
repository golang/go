// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package wycheproof provides helper utilities for writing tests that
// rely on Wycheproof test vector schemas and JSON vector data.
// See https://github.com/C2SP/wycheproof for more information.
package wycheproof

import (
	"crypto"
	"crypto/internal/cryptotest"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"internal/testenv"
	"os"
	"path"
	"reflect"
	"testing"
)

// wycheproofDir, if set, points to a local Wycheproof checkout that
// LoadVectorFile should read test vectors from.
var wycheproofDir = flag.String("wycheproof-dir", "",
	"path to a local Wycheproof checkout to load test vectors from.")

// LoadVectorFile unmarshals Wycheproof JSON test vector file by name.
//
// Typically, the value argument will be a pointer to a Wycheproof schema
// type representing the in-memory structure of the JSON data.
//
// Panics if there is an error reading the Wycheproof JSON vector data file,
// or if it can't be unmarshalled into the provided value.
func LoadVectorFile(t *testing.T, filename string, value any) {
	testenv.SkipIfShortAndSlow(t)

	// We want to avoid a dependency on c2sp/wycheproof or the schema generator
	// in this stdlib code, so we fetch the module at runtime and read the
	// vector JSON from that module clone. The version is pinned to whatever
	// the _schema generator was last run against (see schemaversion.go), so
	// the vectors match the generated schema.go.
	//
	// If -wycheproof-dir is set, read from that local checkout instead, to
	// support testing local updates to Wycheproof.
	dir := *wycheproofDir
	if dir == "" {
		dir = cryptotest.FetchModule(
			t, "github.com/c2sp/wycheproof", wycheproofVersion)
	}

	content, err := os.ReadFile(path.Join(dir, "testvectors_v1", filename))
	if err != nil {
		t.Fatalf("missing Wycheproof vector file %q: %v", filename, err)
	}

	err = json.Unmarshal(content, value)
	if err != nil {
		t.Fatalf("failed to unmarshal vector file %q: %v", filename, err)
	}
}

// ShouldPass returns true if a test should pass informed by expected result
// and flags.
//
// flagsShouldPass is a map used to determine if an "acceptable" result test
// case should pass based on test's flags.
// Every possible flag value that's associated with an "acceptable" result
// should be explicitly specified, otherwise ShouldPass will panic.
func ShouldPass(t *testing.T, result Result, flags []string, flagsShouldPass map[string]bool) bool {
	switch result {
	case "valid":
		return true
	case "invalid":
		return false
	case "acceptable":
		for _, flag := range flags {
			pass, ok := flagsShouldPass[flag]
			if !ok {
				t.Fatalf("unspecified flag: %q", flag)
			}
			if !pass {
				return false
			}
		}
		return true // There are no flags, or all are meant to pass.
	default:
		t.Fatalf("unexpected result: %v", result)
		return false
	}
}

// ParseHash maps from a Wycheproof hash name to a crypto.Hash implementation
// It panics if the provided hash name is unknown.
func ParseHash(h string) crypto.Hash {
	switch h {
	case "SHA-1":
		return crypto.SHA1
	case "SHA-256":
		return crypto.SHA256
	case "SHA-224":
		return crypto.SHA224
	case "SHA-384":
		return crypto.SHA384
	case "SHA-512":
		return crypto.SHA512
	case "SHA-512/224":
		return crypto.SHA512_224
	case "SHA-512/256":
		return crypto.SHA512_256
	case "SHA3-224":
		return crypto.SHA3_224
	case "SHA3-256":
		return crypto.SHA3_256
	case "SHA3-384":
		return crypto.SHA3_384
	case "SHA3-512":
		return crypto.SHA3_512
	default:
		panic(fmt.Sprintf("unknown hash algorithm: %q", h))
	}
}

// TestName returns a t.Run subtest name for a Wycheproof test vector.
func TestName(file string, tv any) string {
	v := reflect.ValueOf(tv)
	if v.Kind() == reflect.Pointer {
		v = v.Elem()
	}
	tcID := v.FieldByName("TcId").Int()
	comment := v.FieldByName("Comment").String()
	name := fmt.Sprintf("%s #%d", file, tcID)
	if comment != "" {
		name += " " + comment
	}
	return name
}

// MustDecodeHex is a helper that decodes the provided string or panics.
//
// Many Wycheproof vector values are hex encoded strings and in a test context
// we don't intend to handle decoding errors gracefully.
func MustDecodeHex(h string) []byte {
	d, err := hex.DecodeString(h)
	if err != nil {
		panic(err)
	}
	return d
}

// MustPanic calls fn and fails the test if fn does not panic.
//
// This is useful for testing that invalid inputs (like incorrect nonce sizes
// for AEAD ciphers) properly trigger panics rather than silently accepting
// the bad input.
func MustPanic(t *testing.T, name string, fn func()) {
	t.Helper()
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("%s: expected panic but didn't get one", name)
		}
	}()
	fn()
}
