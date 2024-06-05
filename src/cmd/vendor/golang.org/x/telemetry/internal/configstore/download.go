// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package configstore abstracts interaction with the telemetry config server.
// Telemetry config (golang.org/x/telemetry/config) is distributed as a go
// module containing go.mod and config.json. Programs that upload collected
// counters download the latest config using `go mod download`. This provides
// verification of downloaded configuration and cacheability.
package configstore

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"

	"golang.org/x/telemetry/internal/telemetry"
)

const (
	ModulePath     = "golang.org/x/telemetry/config"
	configFileName = "config.json"
)

// needNoConsole is used on windows to set the windows.CREATE_NO_WINDOW
// creation flag.
var needNoConsole = func(cmd *exec.Cmd) {}

// Download fetches the requested telemetry UploadConfig using "go mod
// download". If envOverlay is provided, it is appended to the environment used
// for invoking the go command.
//
// The second result is the canonical version of the requested configuration.
func Download(version string, envOverlay []string) (*telemetry.UploadConfig, string, error) {
	if version == "" {
		version = "latest"
	}
	modVer := ModulePath + "@" + version
	var stdout, stderr bytes.Buffer
	cmd := exec.Command("go", "mod", "download", "-json", modVer)
	needNoConsole(cmd)
	cmd.Env = append(os.Environ(), envOverlay...)
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		var info struct {
			Error string
		}
		if err := json.Unmarshal(stdout.Bytes(), &info); err == nil && info.Error != "" {
			return nil, "", fmt.Errorf("failed to download config module: %v", info.Error)
		}
		return nil, "", fmt.Errorf("failed to download config module: %w\n%s", err, &stderr)
	}

	var info struct {
		Dir     string
		Version string
		Error   string
	}
	if err := json.Unmarshal(stdout.Bytes(), &info); err != nil || info.Dir == "" {
		return nil, "", fmt.Errorf("failed to download config module (invalid JSON): %w", err)
	}
	data, err := os.ReadFile(filepath.Join(info.Dir, configFileName))
	if err != nil {
		return nil, "", fmt.Errorf("invalid config module: %w", err)
	}
	cfg := new(telemetry.UploadConfig)
	if err := json.Unmarshal(data, cfg); err != nil {
		return nil, "", fmt.Errorf("invalid config: %w", err)
	}
	return cfg, info.Version, nil
}
