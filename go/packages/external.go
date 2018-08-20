// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file enables an external tool to intercept package requests.
// If the tool is present then its results are used in preference to
// the go list command.

package packages

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os/exec"
	"strings"
)

// findExternalTool returns the file path of a tool that supplies supplies
// the build system package structure, or "" if not found."
// If GOPACKAGESDRIVER is set in the environment findExternalTool returns its
// value, otherwise it searches for a binary named gopackagesdriver on the PATH.
func findExternalDriver(cfg *Config) driver {
	const toolPrefix = "GOPACKAGESDRIVER="
	tool := ""
	for _, env := range cfg.Env {
		if val := strings.TrimPrefix(env, toolPrefix); val != env {
			tool = val
		}
	}
	if tool != "" && tool == "off" {
		return nil
	}
	if tool == "" {
		var err error
		tool, err = exec.LookPath("gopackagesdriver")
		if err != nil {
			return nil
		}
	}
	return func(cfg *Config, words ...string) (*driverResponse, error) {
		buf := new(bytes.Buffer)
		fullargs := []string{
			"list",
			fmt.Sprintf("-test=%t", cfg.Tests),
			fmt.Sprintf("-export=%t", usesExportData(cfg)),
			fmt.Sprintf("-deps=%t", cfg.Mode >= LoadImports),
		}
		for _, f := range cfg.BuildFlags {
			fullargs = append(fullargs, fmt.Sprintf("-buildflag=%v", f))
		}
		fullargs = append(fullargs, "--")
		fullargs = append(fullargs, words...)
		cmd := exec.CommandContext(cfg.Context, tool, fullargs...)
		cmd.Env = cfg.Env
		cmd.Dir = cfg.Dir
		cmd.Stdout = buf
		cmd.Stderr = new(bytes.Buffer)
		if err := cmd.Run(); err != nil {
			return nil, fmt.Errorf("%v: %v: %s", tool, err, cmd.Stderr)
		}
		var response driverResponse
		if err := json.Unmarshal(buf.Bytes(), &response); err != nil {
			return nil, err
		}
		return &response, nil
	}
}
