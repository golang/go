// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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
//
// It first removes any existing values for key, so it is safe to call
// even from within cmdbootstrap.
func setEnv(cmd *exec.Cmd, key, value string) {
	kv := key + "=" + value
	if cmd.Env == nil {
		cmd.Env = os.Environ()
	}

	prefix := kv[:len(key)+1]
	for i, entry := range cmd.Env {
		if strings.HasPrefix(entry, prefix) {
			cmd.Env[i] = kv
			return
		}
	}

	cmd.Env = append(cmd.Env, kv)
}

// unsetEnv sets cmd.Env so that key is not present in the environment.
func unsetEnv(cmd *exec.Cmd, key string) {
	if cmd.Env == nil {
		cmd.Env = os.Environ()
	}

	prefix := key + "="
	for i, entry := range cmd.Env {
		if strings.HasPrefix(entry, prefix) {
			cmd.Env = append(cmd.Env[:i], cmd.Env[i+1:]...)
			return
		}
	}
}
