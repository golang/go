// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package buildcfg

import (
	"os"
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
