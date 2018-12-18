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
	"log"
	"os"
	"os/exec"
	"strings"
	"time"
)

var debug = false

// GetSizes returns the sizes used by the underlying driver with the given parameters.
func GetSizes(ctx context.Context, buildFlags, env []string, dir string, usesExportData bool) (types.Sizes, error) {
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
		return GetSizesGolist(ctx, buildFlags, env, dir, usesExportData)
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

func GetSizesGolist(ctx context.Context, buildFlags, env []string, dir string, usesExportData bool) (types.Sizes, error) {
	args := []string{"list", "-f", "{{context.GOARCH}} {{context.Compiler}}"}
	args = append(args, buildFlags...)
	args = append(args, "--", "unsafe")
	stdout, err := InvokeGo(ctx, env, dir, usesExportData, args...)
	if err != nil {
		return nil, err
	}
	fields := strings.Fields(stdout.String())
	goarch := fields[0]
	compiler := fields[1]
	return types.SizesFor(compiler, goarch), nil
}

// InvokeGo returns the stdout of a go command invocation.
func InvokeGo(ctx context.Context, env []string, dir string, usesExportData bool, args ...string) (*bytes.Buffer, error) {
	if debug {
		defer func(start time.Time) { log.Printf("%s for %v", time.Since(start), cmdDebugStr(env, args...)) }(time.Now())
	}
	stdout := new(bytes.Buffer)
	stderr := new(bytes.Buffer)
	cmd := exec.CommandContext(ctx, "go", args...)
	// On darwin the cwd gets resolved to the real path, which breaks anything that
	// expects the working directory to keep the original path, including the
	// go command when dealing with modules.
	// The Go stdlib has a special feature where if the cwd and the PWD are the
	// same node then it trusts the PWD, so by setting it in the env for the child
	// process we fix up all the paths returned by the go command.
	cmd.Env = append(append([]string{}, env...), "PWD="+dir)
	cmd.Dir = dir
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	if err := cmd.Run(); err != nil {
		exitErr, ok := err.(*exec.ExitError)
		if !ok {
			// Catastrophic error:
			// - executable not found
			// - context cancellation
			return nil, fmt.Errorf("couldn't exec 'go %v': %s %T", args, err, err)
		}

		// Export mode entails a build.
		// If that build fails, errors appear on stderr
		// (despite the -e flag) and the Export field is blank.
		// Do not fail in that case.
		if !usesExportData {
			return nil, fmt.Errorf("go %v: %s: %s", args, exitErr, stderr)
		}
	}

	// As of writing, go list -export prints some non-fatal compilation
	// errors to stderr, even with -e set. We would prefer that it put
	// them in the Package.Error JSON (see https://golang.org/issue/26319).
	// In the meantime, there's nowhere good to put them, but they can
	// be useful for debugging. Print them if $GOPACKAGESPRINTGOLISTERRORS
	// is set.
	if len(stderr.Bytes()) != 0 && os.Getenv("GOPACKAGESPRINTGOLISTERRORS") != "" {
		fmt.Fprintf(os.Stderr, "%s stderr: <<%s>>\n", cmdDebugStr(env, args...), stderr)
	}

	// debugging
	if false {
		fmt.Fprintf(os.Stderr, "%s stdout: <<%s>>\n", cmdDebugStr(env, args...), stdout)
	}

	return stdout, nil
}

func cmdDebugStr(envlist []string, args ...string) string {
	env := make(map[string]string)
	for _, kv := range envlist {
		split := strings.Split(kv, "=")
		k, v := split[0], split[1]
		env[k] = v
	}

	return fmt.Sprintf("GOROOT=%v GOPATH=%v GO111MODULE=%v PWD=%v go %v", env["GOROOT"], env["GOPATH"], env["GO111MODULE"], env["PWD"], args)
}
