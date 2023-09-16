// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modfetch

import (
	"context"
	"fmt"
	"io"
	"sort"
	"strings"

	"cmd/go/internal/gover"
	"cmd/go/internal/modfetch/codehost"
)

// A toolchainRepo is a synthesized repository reporting Go toolchain versions.
// It has path "go" or "toolchain". The "go" repo reports versions like "1.2".
// The "toolchain" repo reports versions like "go1.2".
//
// Note that the repo ONLY reports versions. It does not actually support
// downloading of the actual toolchains. Instead, that is done using
// the regular repo code with "golang.org/toolchain".
// The naming conflict is unfortunate: "golang.org/toolchain"
// should perhaps have been "go.dev/dl", but it's too late.
//
// For clarity, this file refers to golang.org/toolchain as the "DL" repo,
// the one you can actually download.
type toolchainRepo struct {
	path string // either "go" or "toolchain"
	repo Repo   // underlying DL repo
}

func (r *toolchainRepo) ModulePath() string {
	return r.path
}

func (r *toolchainRepo) Versions(ctx context.Context, prefix string) (*Versions, error) {
	// Read DL repo list and convert to "go" or "toolchain" version list.
	versions, err := r.repo.Versions(ctx, "")
	if err != nil {
		return nil, err
	}
	versions.Origin = nil
	var list []string
	have := make(map[string]bool)
	goPrefix := ""
	if r.path == "toolchain" {
		goPrefix = "go"
	}
	for _, v := range versions.List {
		v, ok := dlToGo(v)
		if !ok {
			continue
		}
		if !have[v] {
			have[v] = true
			list = append(list, goPrefix+v)
		}
	}

	// Always include our own version.
	// This means that the development branch of Go 1.21 (say) will allow 'go get go@1.21'
	// even though there are no Go 1.21 releases yet.
	// Once there is a release, 1.21 will be treated as a query matching the latest available release.
	// Before then, 1.21 will be treated as a query that resolves to this entry we are adding (1.21).
	if v := gover.Local(); !have[v] {
		list = append(list, goPrefix+v)
	}

	if r.path == "go" {
		sort.Slice(list, func(i, j int) bool {
			return gover.Compare(list[i], list[j]) < 0
		})
	} else {
		sort.Slice(list, func(i, j int) bool {
			return gover.Compare(gover.FromToolchain(list[i]), gover.FromToolchain(list[j])) < 0
		})
	}
	versions.List = list
	return versions, nil
}

func (r *toolchainRepo) Stat(ctx context.Context, rev string) (*RevInfo, error) {
	// Convert rev to DL version and stat that to make sure it exists.
	// In theory the go@ versions should be like 1.21.0
	// and the toolchain@ versions should be like go1.21.0
	// but people will type the wrong one, and so we accept
	// both and silently correct it to the standard form.
	prefix := ""
	v := rev
	v = strings.TrimPrefix(v, "go")
	if r.path == "toolchain" {
		prefix = "go"
	}

	if !gover.IsValid(v) {
		return nil, fmt.Errorf("invalid %s version %s", r.path, rev)
	}

	// If we're asking about "go" (not "toolchain"), pretend to have
	// all earlier Go versions available without network access:
	// we will provide those ourselves, at least in GOTOOLCHAIN=auto mode.
	if r.path == "go" && gover.Compare(v, gover.Local()) <= 0 {
		return &RevInfo{Version: prefix + v}, nil
	}

	// Similarly, if we're asking about *exactly* the current toolchain,
	// we don't need to access the network to know that it exists.
	if r.path == "toolchain" && v == gover.Local() {
		return &RevInfo{Version: prefix + v}, nil
	}

	if gover.IsLang(v) {
		// We can only use a language (development) version if the current toolchain
		// implements that version, and the two checks above have ruled that out.
		return nil, fmt.Errorf("go language version %s is not a toolchain version", rev)
	}

	// Check that the underlying toolchain exists.
	// We always ask about linux-amd64 because that one
	// has always existed and is likely to always exist in the future.
	// This avoids different behavior validating go versions on different
	// architectures. The eventual download uses the right GOOS-GOARCH.
	info, err := r.repo.Stat(ctx, goToDL(v, "linux", "amd64"))
	if err != nil {
		return nil, err
	}

	// Return the info using the canonicalized rev
	// (toolchain 1.2 => toolchain go1.2).
	return &RevInfo{Version: prefix + v, Time: info.Time}, nil
}

func (r *toolchainRepo) Latest(ctx context.Context) (*RevInfo, error) {
	versions, err := r.Versions(ctx, "")
	if err != nil {
		return nil, err
	}
	var max string
	for _, v := range versions.List {
		if max == "" || gover.ModCompare(r.path, v, max) > 0 {
			max = v
		}
	}
	return r.Stat(ctx, max)
}

func (r *toolchainRepo) GoMod(ctx context.Context, version string) (data []byte, err error) {
	return []byte("module " + r.path + "\n"), nil
}

func (r *toolchainRepo) Zip(ctx context.Context, dst io.Writer, version string) error {
	return fmt.Errorf("invalid use of toolchainRepo: Zip")
}

func (r *toolchainRepo) CheckReuse(ctx context.Context, old *codehost.Origin) error {
	return fmt.Errorf("invalid use of toolchainRepo: CheckReuse")
}

// goToDL converts a Go version like "1.2" to a DL module version like "v0.0.1-go1.2.linux-amd64".
func goToDL(v, goos, goarch string) string {
	return "v0.0.1-go" + v + ".linux-amd64"
}

// dlToGo converts a DL module version like "v0.0.1-go1.2.linux-amd64" to a Go version like "1.2".
func dlToGo(v string) (string, bool) {
	// v0.0.1-go1.19.7.windows-amd64
	// cut v0.0.1-
	_, v, ok := strings.Cut(v, "-")
	if !ok {
		return "", false
	}
	// cut .windows-amd64
	i := strings.LastIndex(v, ".")
	if i < 0 || !strings.Contains(v[i+1:], "-") {
		return "", false
	}
	return strings.TrimPrefix(v[:i], "go"), true
}
