// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.18
// +build go1.18

package hooks

import (
	"context"
	"fmt"

	"golang.org/x/tools/gopls/internal/lsp/source"
	"mvdan.cc/gofumpt/format"
)

func updateGofumpt(options *source.Options) {
	options.GofumptFormat = func(ctx context.Context, langVersion, modulePath string, src []byte) ([]byte, error) {
		fixedVersion, err := fixLangVersion(langVersion)
		if err != nil {
			return nil, err
		}
		return format.Source(src, format.Options{
			LangVersion: fixedVersion,
			ModulePath:  modulePath,
		})
	}
}

// fixLangVersion function cleans the input so that gofumpt doesn't panic. It is
// rather permissive, and accepts version strings that aren't technically valid
// in a go.mod file.
//
// More specifically, it looks for an optional 'v' followed by 1-3
// '.'-separated numbers. The resulting string is stripped of any suffix beyond
// this expected version number pattern.
//
// See also golang/go#61692: gofumpt does not accept the new language versions
// appearing in go.mod files (e.g. go1.21rc3).
func fixLangVersion(input string) (string, error) {
	bad := func() (string, error) {
		return "", fmt.Errorf("invalid language version syntax %q", input)
	}
	if input == "" {
		return input, nil
	}
	i := 0
	if input[0] == 'v' { // be flexible about 'v'
		i++
	}
	// takeDigits consumes ascii numerals 0-9 and reports if at least one was
	// consumed.
	takeDigits := func() bool {
		found := false
		for ; i < len(input) && '0' <= input[i] && input[i] <= '9'; i++ {
			found = true
		}
		return found
	}
	if !takeDigits() { // versions must start with at least one number
		return bad()
	}

	// Accept optional minor and patch versions.
	for n := 0; n < 2; n++ {
		if i < len(input) && input[i] == '.' {
			// Look for minor/patch version.
			i++
			if !takeDigits() {
				i--
				break
			}
		}
	}
	// Accept any suffix.
	return input[:i], nil
}
