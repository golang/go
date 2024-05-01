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
	configModulePath = "golang.org/x/telemetry/config"
	configFileName   = "config.json"
)

// DownloadOption is an option for Download.
type DownloadOption struct {
	// Env holds the environment variables used when downloading the configuration.
	// If nil, the process's environment variables are used.
	Env []string
}

// Download fetches the requested telemetry UploadConfig using "go mod download".
//
// The second result is the canonical version of the requested configuration.
func Download(version string, opts *DownloadOption) (telemetry.UploadConfig, string, error) {
	if version == "" {
		version = "latest"
	}
	if opts == nil {
		opts = &DownloadOption{}
	}
	modVer := configModulePath + "@" + version
	var stdout, stderr bytes.Buffer
	cmd := exec.Command("go", "mod", "download", "-json", modVer)
	cmd.Env = opts.Env
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		var info struct {
			Error string
		}
		if err := json.Unmarshal(stdout.Bytes(), &info); err == nil && info.Error != "" {
			return telemetry.UploadConfig{}, "", fmt.Errorf("failed to download config module: %v", info.Error)
		}
		return telemetry.UploadConfig{}, "", fmt.Errorf("failed to download config module: %w\n%s", err, &stderr)
	}

	var info struct {
		Dir     string
		Version string
		Error   string
	}
	if err := json.Unmarshal(stdout.Bytes(), &info); err != nil || info.Dir == "" {
		return telemetry.UploadConfig{}, "", fmt.Errorf("failed to download config module (invalid JSON): %w", err)
	}
	data, err := os.ReadFile(filepath.Join(info.Dir, configFileName))
	if err != nil {
		return telemetry.UploadConfig{}, "", fmt.Errorf("invalid config module: %w", err)
	}
	var cfg telemetry.UploadConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return telemetry.UploadConfig{}, "", fmt.Errorf("invalid config: %w", err)
	}
	return cfg, info.Version, nil
}
