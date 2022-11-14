// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pkgpath

import (
	"internal/testenv"
	"os"
	"testing"
)

const testEnvName = "GO_PKGPATH_TEST_COMPILER"

// This init function supports TestToSymbolFunc. For simplicity,
// we use the test binary itself as a sample gccgo driver.
// We set an environment variable to specify how it should behave.
func init() {
	switch os.Getenv(testEnvName) {
	case "":
		return
	case "v1":
		os.Stdout.WriteString(`.string	"go.l__ufer.Run"`)
		os.Exit(0)
	case "v2":
		os.Stdout.WriteString(`.string	"go.l..u00e4ufer.Run"`)
		os.Exit(0)
	case "v3":
		os.Stdout.WriteString(`.string	"go_0l_u00e4ufer.Run"`)
		os.Exit(0)
	case "error":
		os.Stdout.WriteString(`unknown string`)
		os.Exit(0)
	}
}

func TestToSymbolFunc(t *testing.T) {
	testenv.MustHaveExec(t)

	const input = "pä世🜃"
	tests := []struct {
		env     string
		fail    bool
		mangled string
	}{
		{
			env:     "v1",
			mangled: "p___",
		},
		{
			env:     "v2",
			mangled: "p..u00e4..u4e16..U0001f703",
		},
		{
			env:     "v3",
			mangled: "p_u00e4_u4e16_U0001f703",
		},
		{
			env:  "error",
			fail: true,
		},
	}

	cmd := os.Args[0]
	tmpdir := t.TempDir()

	defer os.Unsetenv(testEnvName)

	for _, test := range tests {
		t.Run(test.env, func(t *testing.T) {
			os.Setenv(testEnvName, test.env)

			fn, err := ToSymbolFunc(cmd, tmpdir)
			if err != nil {
				if !test.fail {
					t.Errorf("ToSymbolFunc(%q, %q): unexpected error %v", cmd, tmpdir, err)
				}
			} else if test.fail {
				t.Errorf("ToSymbolFunc(%q, %q) succeeded but expected to fail", cmd, tmpdir)
			} else if got, want := fn(input), test.mangled; got != want {
				t.Errorf("ToSymbolFunc(%q, %q)(%q) = %q, want %q", cmd, tmpdir, input, got, want)
			}
		})
	}
}

var symbolTests = []struct {
	input, v1, v2, v3 string
}{
	{
		"",
		"",
		"",
		"",
	},
	{
		"bytes",
		"bytes",
		"bytes",
		"bytes",
	},
	{
		"net/http",
		"net_http",
		"net..z2fhttp",
		"net_1http",
	},
	{
		"golang.org/x/net/http",
		"golang_org_x_net_http",
		"golang.x2eorg..z2fx..z2fnet..z2fhttp",
		"golang_0org_1x_1net_1http",
	},
	{
		"pä世.🜃",
		"p____",
		"p..u00e4..u4e16.x2e..U0001f703",
		"p_u00e4_u4e16_0_U0001f703",
	},
}

func TestV1(t *testing.T) {
	for _, test := range symbolTests {
		if got, want := toSymbolV1(test.input), test.v1; got != want {
			t.Errorf("toSymbolV1(%q) = %q, want %q", test.input, got, want)
		}
	}
}

func TestV2(t *testing.T) {
	for _, test := range symbolTests {
		if got, want := toSymbolV2(test.input), test.v2; got != want {
			t.Errorf("toSymbolV2(%q) = %q, want %q", test.input, got, want)
		}
	}
}

func TestV3(t *testing.T) {
	for _, test := range symbolTests {
		if got, want := toSymbolV3(test.input), test.v3; got != want {
			t.Errorf("toSymbolV3(%q) = %q, want %q", test.input, got, want)
		}
	}
}
