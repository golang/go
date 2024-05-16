// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package load

import (
	"errors"
	"fmt"
	"go/build"
	"internal/godebugs"
	"sort"
	"strconv"
	"strings"

	"cmd/go/internal/gover"
	"cmd/go/internal/modload"
)

var ErrNotGoDebug = errors.New("not //go:debug line")

func ParseGoDebug(text string) (key, value string, err error) {
	if !strings.HasPrefix(text, "//go:debug") {
		return "", "", ErrNotGoDebug
	}
	i := strings.IndexAny(text, " \t")
	if i < 0 {
		if strings.TrimSpace(text) == "//go:debug" {
			return "", "", fmt.Errorf("missing key=value")
		}
		return "", "", ErrNotGoDebug
	}
	k, v, ok := strings.Cut(strings.TrimSpace(text[i:]), "=")
	if !ok {
		return "", "", fmt.Errorf("missing key=value")
	}
	if err := modload.CheckGodebug("//go:debug setting", k, v); err != nil {
		return "", "", err
	}
	return k, v, nil
}

// defaultGODEBUG returns the default GODEBUG setting for the main package p.
// When building a test binary, directives, testDirectives, and xtestDirectives
// list additional directives from the package under test.
func defaultGODEBUG(p *Package, directives, testDirectives, xtestDirectives []build.Directive) string {
	if p.Name != "main" {
		return ""
	}
	goVersion := modload.MainModules.GoVersion()
	if modload.RootMode == modload.NoRoot && p.Module != nil {
		// This is go install pkg@version or go run pkg@version.
		// Use the Go version from the package.
		// If there isn't one, then assume Go 1.20,
		// the last version before GODEBUGs were introduced.
		goVersion = p.Module.GoVersion
		if goVersion == "" {
			goVersion = "1.20"
		}
	}

	var m map[string]string
	for _, g := range modload.MainModules.Godebugs() {
		if m == nil {
			m = make(map[string]string)
		}
		m[g.Key] = g.Value
	}
	for _, list := range [][]build.Directive{p.Internal.Build.Directives, directives, testDirectives, xtestDirectives} {
		for _, d := range list {
			k, v, err := ParseGoDebug(d.Text)
			if err != nil {
				continue
			}
			if m == nil {
				m = make(map[string]string)
			}
			m[k] = v
		}
	}
	if v, ok := m["default"]; ok {
		delete(m, "default")
		v = strings.TrimPrefix(v, "go")
		if gover.IsValid(v) {
			goVersion = v
		}
	}

	defaults := godebugForGoVersion(goVersion)
	if defaults != nil {
		// Apply m on top of defaults.
		for k, v := range m {
			defaults[k] = v
		}
		m = defaults
	}

	var keys []string
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var b strings.Builder
	for _, k := range keys {
		if b.Len() > 0 {
			b.WriteString(",")
		}
		b.WriteString(k)
		b.WriteString("=")
		b.WriteString(m[k])
	}
	return b.String()
}

func godebugForGoVersion(v string) map[string]string {
	if strings.Count(v, ".") >= 2 {
		i := strings.Index(v, ".")
		j := i + 1 + strings.Index(v[i+1:], ".")
		v = v[:j]
	}

	if !strings.HasPrefix(v, "1.") {
		return nil
	}
	n, err := strconv.Atoi(v[len("1."):])
	if err != nil {
		return nil
	}

	def := make(map[string]string)
	for _, info := range godebugs.All {
		if n < info.Changed {
			def[info.Name] = info.Old
		}
	}
	return def
}
