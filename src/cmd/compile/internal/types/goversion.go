// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"fmt"
	"internal/goversion"
	"internal/lazyregexp"
	"log"
	"strconv"

	"cmd/compile/internal/base"
)

// A lang is a language version broken into major and minor numbers.
type lang struct {
	major, minor int
}

// langWant is the desired language version set by the -lang flag.
// If the -lang flag is not set, this is the zero value, meaning that
// any language version is supported.
var langWant lang

// AllowsGoVersion reports whether local package is allowed
// to use Go version major.minor.
func AllowsGoVersion(major, minor int) bool {
	if langWant.major == 0 && langWant.minor == 0 {
		return true
	}
	return langWant.major > major || (langWant.major == major && langWant.minor >= minor)
}

// ParseLangFlag verifies that the -lang flag holds a valid value, and
// exits if not. It initializes data used by AllowsGoVersion.
func ParseLangFlag() {
	if base.Flag.Lang == "" {
		return
	}

	var err error
	langWant, err = parseLang(base.Flag.Lang)
	if err != nil {
		log.Fatalf("invalid value %q for -lang: %v", base.Flag.Lang, err)
	}

	if def := currentLang(); base.Flag.Lang != def {
		defVers, err := parseLang(def)
		if err != nil {
			log.Fatalf("internal error parsing default lang %q: %v", def, err)
		}
		if langWant.major > defVers.major || (langWant.major == defVers.major && langWant.minor > defVers.minor) {
			log.Fatalf("invalid value %q for -lang: max known version is %q", base.Flag.Lang, def)
		}
	}
}

// parseLang parses a -lang option into a langVer.
func parseLang(s string) (lang, error) {
	if s == "go1" { // cmd/go's new spelling of "go1.0" (#65528)
		s = "go1.0"
	}

	matches := goVersionRE.FindStringSubmatch(s)
	if matches == nil {
		return lang{}, fmt.Errorf(`should be something like "go1.12"`)
	}
	major, err := strconv.Atoi(matches[1])
	if err != nil {
		return lang{}, err
	}
	minor, err := strconv.Atoi(matches[2])
	if err != nil {
		return lang{}, err
	}
	return lang{major: major, minor: minor}, nil
}

// currentLang returns the current language version.
func currentLang() string {
	return fmt.Sprintf("go1.%d", goversion.Version)
}

// goVersionRE is a regular expression that matches the valid
// arguments to the -lang flag.
var goVersionRE = lazyregexp.New(`^go([1-9]\d*)\.(0|[1-9]\d*)$`)
