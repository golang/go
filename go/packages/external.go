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

// Driver
type driverRequest struct {
	Command    string            `json:"command"`
	Mode       LoadMode          `json:"mode"`
	Env        []string          `json:"env"`
	BuildFlags []string          `json:"build_flags"`
	Tests      bool              `json:"tests"`
	Overlay    map[string][]byte `json:"overlay"`
}

// findExternalDriver returns the file path of a tool that supplies
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
		req, err := json.Marshal(driverRequest{
			Mode:       cfg.Mode,
			Env:        cfg.Env,
			BuildFlags: cfg.BuildFlags,
			Tests:      cfg.Tests,
			Overlay:    cfg.Overlay,
		})
		if err != nil {
			return nil, fmt.Errorf("failed to encode message to driver tool: %v", err)
		}

		buf := new(bytes.Buffer)
		cmd := exec.CommandContext(cfg.Context, tool, words...)
		cmd.Dir = cfg.Dir
		cmd.Env = cfg.Env
		cmd.Stdin = bytes.NewReader(req)
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
