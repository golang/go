// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.19
// +build !go1.19

package main

import (
	"os"
	"os/exec"
	"strings"
)

// setDir sets cmd.Dir to dir, and also adds PWD=dir to cmd's environment.
func setDir(cmd *exec.Cmd, dir string) {
	cmd.Dir = dir
	setEnv(cmd, "PWD", dir)
}

// setEnv sets cmd.Env so that key = value.
func setEnv(cmd *exec.Cmd, key, value string) {
	kv := key + "=" + value
	if cmd.Env == nil {
		cmd.Env = os.Environ()
	}
	cmd.Env = append(cmd.Env, kv)
}

// unsetEnv sets cmd.Env so that key is not present in the environment.
func unsetEnv(cmd *exec.Cmd, key string) {
	if cmd.Env == nil {
		cmd.Env = os.Environ()
	}

	prefix := key + "="
	newEnv := []string{}
	for _, entry := range cmd.Env {
		if strings.HasPrefix(entry, prefix) {
			continue
		}
		newEnv = append(newEnv, entry)
		// key may appear multiple times, so keep going.
	}
	cmd.Env = newEnv
}
