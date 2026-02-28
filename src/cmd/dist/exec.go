// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"os/exec"
	"strings"
)

// setDir sets cmd.Dir to dir, and also adds PWD=dir to cmd's environment.
func setDir(cmd *exec.Cmd, dir string) {
	cmd.Dir = dir
	if cmd.Env != nil {
		// os/exec won't set PWD automatically.
		setEnv(cmd, "PWD", dir)
	}
}

// setEnv sets cmd.Env so that key = value.
func setEnv(cmd *exec.Cmd, key, value string) {
	cmd.Env = append(cmd.Environ(), key+"="+value)
}

// unsetEnv sets cmd.Env so that key is not present in the environment.
func unsetEnv(cmd *exec.Cmd, key string) {
	cmd.Env = cmd.Environ()

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
