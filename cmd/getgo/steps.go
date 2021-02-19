// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !plan9
// +build !plan9

package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

type step func(context.Context) error

func welcome(ctx context.Context) error {
	fmt.Println("Welcome to the Go installer!")
	answer, err := prompt(ctx, "Would you like to install Go? Y/n", "Y")
	if err != nil {
		return err
	}
	if strings.ToLower(answer) != "y" {
		fmt.Println("Exiting install.")
		return errExitCleanly
	}

	return nil
}

func checkOthers(ctx context.Context) error {
	// TODO: if go is currently installed install new version over that
	path, err := whichGo(ctx)
	if err != nil {
		fmt.Printf("Cannot check if Go is already installed:\n%v\n", err)
	}
	if path == "" {
		return nil
	}
	if path != installPath {
		fmt.Printf("Go is already installed at %v; remove it from your PATH.\n", path)
	}
	return nil
}

func chooseVersion(ctx context.Context) error {
	if *goVersion != "" {
		return nil
	}

	var err error
	*goVersion, err = getLatestGoVersion()
	if err != nil {
		return err
	}

	answer, err := prompt(ctx, fmt.Sprintf("The latest Go version is %s, install that? Y/n", *goVersion), "Y")
	if err != nil {
		return err
	}

	if strings.ToLower(answer) != "y" {
		// TODO: handle passing a version
		fmt.Println("Aborting install.")
		return errExitCleanly
	}

	return nil
}

func downloadGo(ctx context.Context) error {
	answer, err := prompt(ctx, fmt.Sprintf("Download Go version %s to %s? Y/n", *goVersion, installPath), "Y")
	if err != nil {
		return err
	}

	if strings.ToLower(answer) != "y" {
		fmt.Println("Aborting install.")
		return errExitCleanly
	}

	fmt.Printf("Downloading Go version %s to %s\n", *goVersion, installPath)
	fmt.Println("This may take a bit of time...")

	if err := downloadGoVersion(*goVersion, runtime.GOOS, arch, installPath); err != nil {
		return err
	}

	if err := appendToPATH(filepath.Join(installPath, "bin")); err != nil {
		return err
	}

	fmt.Println("Downloaded!")
	return nil
}

func setupGOPATH(ctx context.Context) error {
	answer, err := prompt(ctx, "Would you like us to setup your GOPATH? Y/n", "Y")
	if err != nil {
		return err
	}

	if strings.ToLower(answer) != "y" {
		fmt.Println("Exiting and not setting up GOPATH.")
		return errExitCleanly
	}

	fmt.Println("Setting up GOPATH")
	home, err := getHomeDir()
	if err != nil {
		return err
	}

	gopath := os.Getenv("GOPATH")
	if gopath == "" {
		// set $GOPATH
		gopath = filepath.Join(home, "go")
		if err := persistEnvVar("GOPATH", gopath); err != nil {
			return err
		}
		fmt.Println("GOPATH has been set up!")
	} else {
		verbosef("GOPATH is already set to %s", gopath)
	}

	if err := appendToPATH(filepath.Join(gopath, "bin")); err != nil {
		return err
	}
	return persistEnvChangesForSession()
}
