// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package packagesdriver fetches type sizes for go/packages and go/analysis.
package packagesdriver

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"go/types"
	"os/exec"
	"strings"

	"golang.org/x/tools/internal/gocommand"
)

var debug = false

func GetSizes(ctx context.Context, buildFlags, env []string, gocmdRunner *gocommand.Runner, dir string) (types.Sizes, error) {
	// TODO(matloob): Clean this up. This code is mostly a copy of packages.findExternalDriver.
	const toolPrefix = "GOPACKAGESDRIVER="
	tool := ""
	for _, env := range env {
		if val := strings.TrimPrefix(env, toolPrefix); val != env {
			tool = val
		}
	}

	if tool == "" {
		var err error
		tool, err = exec.LookPath("gopackagesdriver")
		if err != nil {
			// We did not find the driver, so use "go list".
			tool = "off"
		}
	}

	if tool == "off" {
		inv := gocommand.Invocation{
			BuildFlags: buildFlags,
			Env:        env,
			WorkingDir: dir,
		}
		return GetSizesGolist(ctx, inv, gocmdRunner)
	}

	req, err := json.Marshal(struct {
		Command    string   `json:"command"`
		Env        []string `json:"env"`
		BuildFlags []string `json:"build_flags"`
	}{
		Command:    "sizes",
		Env:        env,
		BuildFlags: buildFlags,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to encode message to driver tool: %v", err)
	}

	buf := new(bytes.Buffer)
	cmd := exec.CommandContext(ctx, tool)
	cmd.Dir = dir
	cmd.Env = env
	cmd.Stdin = bytes.NewReader(req)
	cmd.Stdout = buf
	cmd.Stderr = new(bytes.Buffer)
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("%v: %v: %s", tool, err, cmd.Stderr)
	}
	var response struct {
		// Sizes, if not nil, is the types.Sizes to use when type checking.
		Sizes *types.StdSizes
	}
	if err := json.Unmarshal(buf.Bytes(), &response); err != nil {
		return nil, err
	}
	return response.Sizes, nil
}

func GetSizesGolist(ctx context.Context, inv gocommand.Invocation, gocmdRunner *gocommand.Runner) (types.Sizes, error) {
	inv.Verb = "list"
	inv.Args = []string{"-f", "{{context.GOARCH}} {{context.Compiler}}", "--", "unsafe"}
	stdout, stderr, friendlyErr, rawErr := gocmdRunner.RunRaw(ctx, inv)
	var goarch, compiler string
	if rawErr != nil {
		if strings.Contains(rawErr.Error(), "cannot find main module") {
			// User's running outside of a module. All bets are off. Get GOARCH and guess compiler is gc.
			// TODO(matloob): Is this a problem in practice?
			inv.Verb = "env"
			inv.Args = []string{"GOARCH"}
			envout, enverr := gocmdRunner.Run(ctx, inv)
			if enverr != nil {
				return nil, enverr
			}
			goarch = strings.TrimSpace(envout.String())
			compiler = "gc"
		} else {
			return nil, friendlyErr
		}
	} else {
		fields := strings.Fields(stdout.String())
		if len(fields) < 2 {
			return nil, fmt.Errorf("could not parse GOARCH and Go compiler in format \"<GOARCH> <compiler>\":\nstdout: <<%s>>\nstderr: <<%s>>",
				stdout.String(), stderr.String())
		}
		goarch = fields[0]
		compiler = fields[1]
	}
	return types.SizesFor(compiler, goarch), nil
}
