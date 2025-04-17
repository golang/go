// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package toolchain

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/gover"
	"cmd/go/internal/modfetch"
	"cmd/internal/telemetry/counter"
)

// A Switcher collects errors to be reported and then decides
// between reporting the errors or switching to a new toolchain
// to resolve them.
//
// The client calls [Switcher.Error] repeatedly with errors encountered
// and then calls [Switcher.Switch]. If the errors included any
// *gover.TooNewErrors (potentially wrapped) and switching is
// permitted by GOTOOLCHAIN, Switch switches to a new toolchain.
// Otherwise Switch prints all the errors using base.Error.
//
// See https://go.dev/doc/toolchain#switch.
type Switcher struct {
	TooNew *gover.TooNewError // max go requirement observed
	Errors []error            // errors collected so far
}

// Error reports the error to the Switcher,
// which saves it for processing during Switch.
func (s *Switcher) Error(err error) {
	s.Errors = append(s.Errors, err)
	s.addTooNew(err)
}

// addTooNew adds any TooNew errors that can be found in err.
func (s *Switcher) addTooNew(err error) {
	switch err := err.(type) {
	case interface{ Unwrap() []error }:
		for _, e := range err.Unwrap() {
			s.addTooNew(e)
		}

	case interface{ Unwrap() error }:
		s.addTooNew(err.Unwrap())

	case *gover.TooNewError:
		if s.TooNew == nil ||
			gover.Compare(err.GoVersion, s.TooNew.GoVersion) > 0 ||
			gover.Compare(err.GoVersion, s.TooNew.GoVersion) == 0 && err.What < s.TooNew.What {
			s.TooNew = err
		}
	}
}

// NeedSwitch reports whether Switch would attempt to switch toolchains.
func (s *Switcher) NeedSwitch() bool {
	return s.TooNew != nil && (HasAuto() || HasPath())
}

// Switch decides whether to switch to a newer toolchain
// to resolve any of the saved errors.
// It switches if toolchain switches are permitted and there is at least one TooNewError.
//
// If Switch decides not to switch toolchains, it prints the errors using base.Error and returns.
//
// If Switch decides to switch toolchains but cannot identify a toolchain to use.
// it prints the errors along with one more about not being able to find the toolchain
// and returns.
//
// Otherwise, Switch prints an informational message giving a reason for the
// switch and the toolchain being invoked and then switches toolchains.
// This operation never returns.
func (s *Switcher) Switch(ctx context.Context) {
	if !s.NeedSwitch() {
		for _, err := range s.Errors {
			base.Error(err)
		}
		return
	}

	// Switch to newer Go toolchain if necessary and possible.
	tv, err := NewerToolchain(ctx, s.TooNew.GoVersion)
	if err != nil {
		for _, err := range s.Errors {
			base.Error(err)
		}
		base.Error(fmt.Errorf("switching to go >= %v: %w", s.TooNew.GoVersion, err))
		return
	}

	fmt.Fprintf(os.Stderr, "go: %v requires go >= %v; switching to %v\n", s.TooNew.What, s.TooNew.GoVersion, tv)
	counterSwitchExec.Inc()
	Exec(tv)
	panic("unreachable")
}

var counterSwitchExec = counter.New("go/toolchain/switch-exec")

// SwitchOrFatal attempts a toolchain switch based on the information in err
// and otherwise falls back to base.Fatal(err).
func SwitchOrFatal(ctx context.Context, err error) {
	var s Switcher
	s.Error(err)
	s.Switch(ctx)
	base.Exit()
}

// NewerToolchain returns the name of the toolchain to use when we need
// to switch to a newer toolchain that must support at least the given Go version.
// See https://go.dev/doc/toolchain#switch.
//
// If the latest major release is 1.N.0, we use the latest patch release of 1.(N-1) if that's >= version.
// Otherwise we use the latest 1.N if that's allowed.
// Otherwise we use the latest release.
func NewerToolchain(ctx context.Context, version string) (string, error) {
	fetch := autoToolchains
	if !HasAuto() {
		fetch = pathToolchains
	}
	list, err := fetch(ctx)
	if err != nil {
		return "", err
	}
	return newerToolchain(version, list)
}

// autoToolchains returns the list of toolchain versions available to GOTOOLCHAIN=auto or =min+auto mode.
func autoToolchains(ctx context.Context) ([]string, error) {
	var versions *modfetch.Versions
	err := modfetch.TryProxies(func(proxy string) error {
		v, err := modfetch.Lookup(ctx, proxy, "go").Versions(ctx, "")
		if err != nil {
			return err
		}
		versions = v
		return nil
	})
	if err != nil {
		return nil, err
	}
	return versions.List, nil
}

// pathToolchains returns the list of toolchain versions available to GOTOOLCHAIN=path or =min+path mode.
func pathToolchains(ctx context.Context) ([]string, error) {
	have := make(map[string]bool)
	var list []string
	for _, dir := range pathDirs() {
		if dir == "" || !filepath.IsAbs(dir) {
			// Refuse to use local directories in $PATH (hard-coding exec.ErrDot).
			continue
		}
		entries, err := os.ReadDir(dir)
		if err != nil {
			continue
		}
		for _, de := range entries {
			if de.IsDir() || !strings.HasPrefix(de.Name(), "go1.") {
				continue
			}
			info, err := de.Info()
			if err != nil {
				continue
			}
			v, ok := pathVersion(dir, de, info)
			if !ok || !strings.HasPrefix(v, "1.") || have[v] {
				continue
			}
			have[v] = true
			list = append(list, v)
		}
	}
	sort.Slice(list, func(i, j int) bool {
		return gover.Compare(list[i], list[j]) < 0
	})
	return list, nil
}

// newerToolchain implements NewerToolchain where the list of choices is known.
// It is separated out for easier testing of this logic.
func newerToolchain(need string, list []string) (string, error) {
	// Consider each release in the list, from newest to oldest,
	// considering only entries >= need and then only entries
	// that are the latest in their language family
	// (the latest 1.40, the latest 1.39, and so on).
	// We prefer the latest patch release before the most recent release family,
	// so if the latest release is 1.40.1 we'll take the latest 1.39.X.
	// Failing that, we prefer the latest patch release before the most recent
	// prerelease family, so if the latest release is 1.40rc1 is out but 1.39 is okay,
	// we'll still take 1.39.X.
	// Failing that we'll take the latest release.
	latest := ""
	for i := len(list) - 1; i >= 0; i-- {
		v := list[i]
		if gover.Compare(v, need) < 0 {
			break
		}
		if gover.Lang(latest) == gover.Lang(v) {
			continue
		}
		newer := latest
		latest = v
		if newer != "" && !gover.IsPrerelease(newer) {
			// latest is the last patch release of Go 1.X, and we saw a non-prerelease of Go 1.(X+1),
			// so latest is the one we want.
			break
		}
	}
	if latest == "" {
		return "", fmt.Errorf("no releases found for go >= %v", need)
	}
	return "go" + latest, nil
}

// HasAuto reports whether the GOTOOLCHAIN setting allows "auto" upgrades.
func HasAuto() bool {
	env := cfg.Getenv("GOTOOLCHAIN")
	return env == "auto" || strings.HasSuffix(env, "+auto")
}

// HasPath reports whether the GOTOOLCHAIN setting allows "path" upgrades.
func HasPath() bool {
	env := cfg.Getenv("GOTOOLCHAIN")
	return env == "path" || strings.HasSuffix(env, "+path")
}
