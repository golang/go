// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin || dragonfly || freebsd || linux || nacl || netbsd || openbsd || solaris
// +build aix darwin dragonfly freebsd linux nacl netbsd openbsd solaris

package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
)

const (
	envSeparator = ":"
	homeKey      = "HOME"
	lineEnding   = "\n"
	pathVar      = "$PATH"
)

var installPath = func() string {
	home, err := getHomeDir()
	if err != nil {
		return "/usr/local/go"
	}

	return filepath.Join(home, ".go")
}()

func whichGo(ctx context.Context) (string, error) {
	return findGo(ctx, "which")
}

func isWindowsXP() bool {
	return false
}

func currentShell() string {
	return os.Getenv("SHELL")
}

func persistEnvChangesForSession() error {
	shellConfig, err := shellConfigFile()
	if err != nil {
		return err
	}
	fmt.Println()
	fmt.Printf("One more thing! Run `source %s` to persist the\n", shellConfig)
	fmt.Println("new environment variables to your current session, or open a")
	fmt.Println("new shell prompt.")

	return nil
}
