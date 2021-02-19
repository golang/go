// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !plan9
// +build !plan9

package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"os/user"
	"path/filepath"
	"runtime"
	"strings"
)

const (
	bashConfig = ".bash_profile"
	zshConfig  = ".zshrc"
)

// appendToPATH adds the given path to the PATH environment variable and
// persists it for future sessions.
func appendToPATH(value string) error {
	if isInPATH(value) {
		return nil
	}
	return persistEnvVar("PATH", pathVar+envSeparator+value)
}

func isInPATH(dir string) bool {
	p := os.Getenv("PATH")

	paths := strings.Split(p, envSeparator)
	for _, d := range paths {
		if d == dir {
			return true
		}
	}

	return false
}

func getHomeDir() (string, error) {
	home := os.Getenv(homeKey)
	if home != "" {
		return home, nil
	}

	u, err := user.Current()
	if err != nil {
		return "", err
	}
	return u.HomeDir, nil
}

func checkStringExistsFile(filename, value string) (bool, error) {
	file, err := os.OpenFile(filename, os.O_RDONLY, 0600)
	if err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		if line == value {
			return true, nil
		}
	}

	return false, scanner.Err()
}

func appendToFile(filename, value string) error {
	verbosef("Adding %q to %s", value, filename)

	ok, err := checkStringExistsFile(filename, value)
	if err != nil {
		return err
	}
	if ok {
		// Nothing to do.
		return nil
	}

	f, err := os.OpenFile(filename, os.O_APPEND|os.O_WRONLY|os.O_CREATE, 0600)
	if err != nil {
		return err
	}
	defer f.Close()

	_, err = f.WriteString(lineEnding + value + lineEnding)
	return err
}

func isShell(name string) bool {
	return strings.Contains(currentShell(), name)
}

// persistEnvVarWindows sets an environment variable in the Windows
// registry.
func persistEnvVarWindows(name, value string) error {
	_, err := runCommand(context.Background(), "powershell", "-command",
		fmt.Sprintf(`[Environment]::SetEnvironmentVariable("%s", "%s", "User")`, name, value))
	return err
}

func persistEnvVar(name, value string) error {
	if runtime.GOOS == "windows" {
		if err := persistEnvVarWindows(name, value); err != nil {
			return err
		}

		if isShell("cmd.exe") || isShell("powershell.exe") {
			return os.Setenv(strings.ToUpper(name), value)
		}
		// User is in bash, zsh, etc.
		// Also set the environment variable in their shell config.
	}

	rc, err := shellConfigFile()
	if err != nil {
		return err
	}

	line := fmt.Sprintf("export %s=%s", strings.ToUpper(name), value)
	if err := appendToFile(rc, line); err != nil {
		return err
	}

	return os.Setenv(strings.ToUpper(name), value)
}

func shellConfigFile() (string, error) {
	home, err := getHomeDir()
	if err != nil {
		return "", err
	}

	switch {
	case isShell("bash"):
		return filepath.Join(home, bashConfig), nil
	case isShell("zsh"):
		return filepath.Join(home, zshConfig), nil
	default:
		return "", fmt.Errorf("%q is not a supported shell", currentShell())
	}
}
