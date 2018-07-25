// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file enables an external tool to intercept package requests.
// If the tool is present then its results are used in preference to
// the go list command.

package packages

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"strings"

	"golang.org/x/tools/go/packages/raw"
)

// externalPackages uses an external command to interpret the words and produce
// raw packages.
// dir may be "" and env may be nil, as per os/exec.Command.
func findRawTool(ctx context.Context, cfg *raw.Config) string {
	const toolPrefix = "GOPACKAGESRAW="
	for _, env := range cfg.Env {
		if val := strings.TrimPrefix(env, toolPrefix); val != env {
			return val
		}
	}
	if found, err := exec.LookPath("gopackagesraw"); err == nil {
		return found
	}
	return ""
}

// externalPackages uses an external command to interpret the words and produce
// raw packages.
// cfg.Dir may be "" and cfg.Env may be nil, as per os/exec.Command.
func externalPackages(ctx context.Context, cfg *raw.Config, tool string, words ...string) ([]string, []*raw.Package, error) {
	buf := new(bytes.Buffer)
	fullargs := []string{
		fmt.Sprintf("-test=%t", cfg.Tests),
		fmt.Sprintf("-export=%t", cfg.Export),
		fmt.Sprintf("-deps=%t", cfg.Deps),
	}
	for _, f := range cfg.Flags {
		fullargs = append(fullargs, fmt.Sprintf("-flags=%v", f))
	}
	fullargs = append(fullargs, "--")
	fullargs = append(fullargs, words...)
	cmd := exec.CommandContext(ctx, tool, fullargs...)
	cmd.Env = cfg.Env
	cmd.Dir = cfg.Dir
	cmd.Stdout = buf
	cmd.Stderr = new(bytes.Buffer)
	if err := cmd.Run(); err != nil {
		return nil, nil, fmt.Errorf("%v: %v: %s", tool, err, cmd.Stderr)
	}
	var results raw.Results
	var pkgs []*raw.Package
	dec := json.NewDecoder(buf)
	if err := dec.Decode(&results); err != nil {
		return nil, nil, fmt.Errorf("JSON decoding raw.Results failed: %v", err)
	}
	if results.Error != "" {
		return nil, nil, errors.New(results.Error)
	}
	for dec.More() {
		p := new(raw.Package)
		if err := dec.Decode(p); err != nil {
			return nil, nil, fmt.Errorf("JSON decoding raw.Package failed: %v", err)
		}
		pkgs = append(pkgs, p)
	}
	return results.Roots, pkgs, nil
}
