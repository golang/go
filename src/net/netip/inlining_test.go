// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package netip

import (
	"internal/testenv"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"
)

func TestInlining(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	t.Parallel()
	var exe string
	if runtime.GOOS == "windows" {
		exe = ".exe"
	}
	out, err := exec.Command(
		filepath.Join(runtime.GOROOT(), "bin", "go"+exe),
		"build",
		"--gcflags=-m",
		"net/netip").CombinedOutput()
	if err != nil {
		t.Fatalf("go build: %v, %s", err, out)
	}
	got := map[string]bool{}
	regexp.MustCompile(` can inline (\S+)`).ReplaceAllFunc(out, func(match []byte) []byte {
		got[strings.TrimPrefix(string(match), " can inline ")] = true
		return nil
	})
	wantInlinable := []string{
		"(*uint128).halves",
		"Addr.BitLen",
		"Addr.hasZone",
		"Addr.Is4",
		"Addr.Is4In6",
		"Addr.Is6",
		"Addr.IsLoopback",
		"Addr.IsMulticast",
		"Addr.IsInterfaceLocalMulticast",
		"Addr.IsValid",
		"Addr.IsUnspecified",
		"Addr.Less",
		"Addr.lessOrEq",
		"Addr.Unmap",
		"Addr.Zone",
		"Addr.v4",
		"Addr.v6",
		"Addr.v6u16",
		"Addr.withoutZone",
		"AddrPortFrom",
		"AddrPort.Addr",
		"AddrPort.Port",
		"AddrPort.IsValid",
		"Prefix.IsSingleIP",
		"Prefix.Masked",
		"Prefix.IsValid",
		"PrefixFrom",
		"Prefix.Addr",
		"Prefix.Bits",
		"AddrFrom4",
		"IPv6LinkLocalAllNodes",
		"IPv6Unspecified",
		"MustParseAddr",
		"MustParseAddrPort",
		"MustParsePrefix",
		"appendDecimal",
		"appendHex",
		"uint128.addOne",
		"uint128.and",
		"uint128.bitsClearedFrom",
		"uint128.bitsSetFrom",
		"uint128.isZero",
		"uint128.not",
		"uint128.or",
		"uint128.subOne",
		"uint128.xor",
	}
	switch runtime.GOARCH {
	case "amd64", "arm64":
		// These don't inline on 32-bit.
		wantInlinable = append(wantInlinable,
			"u64CommonPrefixLen",
			"uint128.commonPrefixLen",
			"Addr.Next",
			"Addr.Prev",
		)
	}

	for _, want := range wantInlinable {
		if !got[want] {
			t.Errorf("%q is no longer inlinable", want)
			continue
		}
		delete(got, want)
	}
	for sym := range got {
		if strings.Contains(sym, ".func") {
			continue
		}
		t.Logf("not in expected set, but also inlinable: %q", sym)

	}
}
