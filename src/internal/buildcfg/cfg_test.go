// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package buildcfg

import (
	"os"
	"reflect"
	"strings"
	"testing"
)

func TestConfigFlags(t *testing.T) {
	os.Setenv("GOAMD64", "v1")
	if goamd64() != 1 {
		t.Errorf("Wrong parsing of GOAMD64=v1")
	}
	os.Setenv("GOAMD64", "v4")
	if goamd64() != 4 {
		t.Errorf("Wrong parsing of GOAMD64=v4")
	}
	Error = nil
	os.Setenv("GOAMD64", "1")
	if goamd64(); Error == nil {
		t.Errorf("Wrong parsing of GOAMD64=1")
	}

	os.Setenv("GORISCV64", "rva20u64")
	if goriscv64() != 20 {
		t.Errorf("Wrong parsing of RISCV64=rva20u64")
	}
	os.Setenv("GORISCV64", "rva22u64")
	if goriscv64() != 22 {
		t.Errorf("Wrong parsing of RISCV64=rva22u64")
	}
	os.Setenv("GORISCV64", "rva23u64")
	if goriscv64() != 23 {
		t.Errorf("Wrong parsing of RISCV64=rva23u64")
	}
	Error = nil
	os.Setenv("GORISCV64", "rva22")
	if _ = goriscv64(); Error == nil {
		t.Errorf("Wrong parsing of RISCV64=rva22")
	}
	Error = nil
	os.Setenv("GOARM64", "v7.0")
	if _ = goarm64(); Error == nil {
		t.Errorf("Wrong parsing of GOARM64=7.0")
	}
	Error = nil
	os.Setenv("GOARM64", "8.0")
	if _ = goarm64(); Error == nil {
		t.Errorf("Wrong parsing of GOARM64=8.0")
	}
	Error = nil
	os.Setenv("GOARM64", "v8.0,lsb")
	if _ = goarm64(); Error == nil {
		t.Errorf("Wrong parsing of GOARM64=v8.0,lsb")
	}
	os.Setenv("GOARM64", "v8.0,lse")
	if goarm64().Version != "v8.0" || goarm64().LSE != true || goarm64().Crypto != false {
		t.Errorf("Wrong parsing of GOARM64=v8.0,lse")
	}
	os.Setenv("GOARM64", "v8.0,crypto")
	if goarm64().Version != "v8.0" || goarm64().LSE != false || goarm64().Crypto != true {
		t.Errorf("Wrong parsing of GOARM64=v8.0,crypto")
	}
	os.Setenv("GOARM64", "v8.0,crypto,lse")
	if goarm64().Version != "v8.0" || goarm64().LSE != true || goarm64().Crypto != true {
		t.Errorf("Wrong parsing of GOARM64=v8.0,crypto,lse")
	}
	os.Setenv("GOARM64", "v8.0,lse,crypto")
	if goarm64().Version != "v8.0" || goarm64().LSE != true || goarm64().Crypto != true {
		t.Errorf("Wrong parsing of GOARM64=v8.0,lse,crypto")
	}
	os.Setenv("GOARM64", "v9.0")
	if goarm64().Version != "v9.0" || goarm64().LSE != true || goarm64().Crypto != false {
		t.Errorf("Wrong parsing of GOARM64=v9.0")
	}
}

func TestParseGORISCV64Valid(t *testing.T) {
	profile, ext, err := ParseGORISCV64("rva23u64,zacas,zabha")
	if err != nil {
		t.Fatalf("ParseGORISCV64 returned error: %v", err)
	}
	if profile != "rva23u64" {
		t.Fatalf("profile = %q, want %q", profile, "rva23u64")
	}
	want := map[string]bool{Riscv64ExtZacas: true, Riscv64ExtZabha: true}
	if !reflect.DeepEqual(ext, want) {
		t.Fatalf("extensions = %#v, want %#v", ext, want)
	}
}

func TestParseGORISCV64InvalidExtension(t *testing.T) {
	_, _, err := ParseGORISCV64("rva23u64,foo")
	if err == nil {
		t.Fatalf("expected error for invalid extension")
	}
	if !strings.Contains(err.Error(), "invalid GORISCV64 extension") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestAllowedRiscv64OptListTrueOnly(t *testing.T) {
	orig := allowedRiscv64Opt
	t.Cleanup(func() { allowedRiscv64Opt = orig })

	allowedRiscv64Opt = map[string]bool{
		Riscv64ExtZacas: true,
		"zfake":         false,
	}
	list := allowedRiscv64OptList()
	if list != Riscv64ExtZacas {
		t.Fatalf("allowedRiscv64OptList() = %q, want %q", list, Riscv64ExtZacas)
	}
}

func TestGoriscv64Extensions(t *testing.T) {
	t.Setenv("GORISCV64", "rva23u64,zacas")
	ext := goriscv64Extensions()
	want := map[string]bool{Riscv64ExtZacas: true}
	if !reflect.DeepEqual(ext, want) {
		t.Fatalf("extensions = %#v, want %#v", ext, want)
	}
}

func TestGoriscv64ExtensionsInvalid(t *testing.T) {
	t.Setenv("GORISCV64", "rva23u64,foo")
	ext := goriscv64Extensions()
	if len(ext) != 0 {
		t.Fatalf("extensions = %#v, want empty map on invalid extension", ext)
	}
}

func TestGoriscv64ExtensionsCaseInsensitive(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  map[string]bool
	}{
		{
			name:  "all lowercase",
			input: "rva23u64,zacas,zabha",
			want:  map[string]bool{Riscv64ExtZacas: true, Riscv64ExtZabha: true},
		},
		{
			name:  "all uppercase",
			input: "rva23u64,ZACAS,ZABHA",
			want:  map[string]bool{Riscv64ExtZacas: true, Riscv64ExtZabha: true},
		},
		{
			name:  "mixed case",
			input: "rva23u64,ZaCaS,zAbHa",
			want:  map[string]bool{Riscv64ExtZacas: true, Riscv64ExtZabha: true},
		},
		{
			name:  "first uppercase",
			input: "rva23u64,Zacas,Zabha",
			want:  map[string]bool{Riscv64ExtZacas: true, Riscv64ExtZabha: true},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			profile, ext, err := ParseGORISCV64(tt.input)
			if err != nil {
				t.Fatalf("ParseGORISCV64 returned error: %v", err)
			}
			if profile != "rva23u64" {
				t.Fatalf("profile = %q, want %q", profile, "rva23u64")
			}
			if !reflect.DeepEqual(ext, tt.want) {
				t.Fatalf("extensions = %#v, want %#v", ext, tt.want)
			}
			// Verify all keys are lowercase
			for k := range ext {
				if k != strings.ToLower(k) {
					t.Errorf("extension key %q is not lowercase", k)
				}
			}
		})
	}
}

func TestGoarm64FeaturesSupports(t *testing.T) {
	g, _ := ParseGoarm64("v9.3")

	if !g.Supports("v9.3") {
		t.Errorf("Wrong goarm64Features.Supports for v9.3, v9.3")
	}

	if g.Supports("v9.4") {
		t.Errorf("Wrong goarm64Features.Supports for v9.3, v9.4")
	}

	if !g.Supports("v8.8") {
		t.Errorf("Wrong goarm64Features.Supports for v9.3, v8.8")
	}

	if g.Supports("v8.9") {
		t.Errorf("Wrong goarm64Features.Supports for v9.3, v8.9")
	}

	if g.Supports(",lse") {
		t.Errorf("Wrong goarm64Features.Supports for v9.3, ,lse")
	}
}

func TestGogoarchTags(t *testing.T) {
	old_goarch := GOARCH
	old_goarm64 := GOARM64

	GOARCH = "arm64"

	os.Setenv("GOARM64", "v9.5")
	GOARM64 = goarm64()
	tags := gogoarchTags()
	want := []string{"arm64.v9.0", "arm64.v9.1", "arm64.v9.2", "arm64.v9.3", "arm64.v9.4", "arm64.v9.5",
		"arm64.v8.0", "arm64.v8.1", "arm64.v8.2", "arm64.v8.3", "arm64.v8.4", "arm64.v8.5", "arm64.v8.6", "arm64.v8.7", "arm64.v8.8", "arm64.v8.9"}
	if len(tags) != len(want) {
		t.Errorf("Wrong number of tags for GOARM64=v9.5")
	} else {
		for i, v := range tags {
			if v != want[i] {
				t.Error("Wrong tags for GOARM64=v9.5")
				break
			}
		}
	}

	GOARCH = old_goarch
	GOARM64 = old_goarm64
}

var goodFIPS = []string{
	"v1.0.0",
	"v1.0.1",
	"v1.2.0",
	"v1.2.3",
}

var badFIPS = []string{
	"v1.0.0-fips",
	"v1.0.0+fips",
	"1.0.0",
	"x1.0.0",
}

func TestIsFIPSVersion(t *testing.T) {
	// good
	for _, s := range goodFIPS {
		if !isFIPSVersion(s) {
			t.Errorf("isFIPSVersion(%q) = false, want true", s)
		}
	}
	// truncated
	const v = "v1.2.3"
	for i := 0; i < len(v); i++ {
		if isFIPSVersion(v[:i]) {
			t.Errorf("isFIPSVersion(%q) = true, want false", v[:i])
		}
	}
	// bad
	for _, s := range badFIPS {
		if isFIPSVersion(s) {
			t.Errorf("isFIPSVersion(%q) = true, want false", s)
		}
	}
}
