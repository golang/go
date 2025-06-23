// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (!cgo && !darwin && !windows && !plan9) || android || (osusergo && !windows && !plan9)

package user

import (
	"fmt"
	"os"
	"runtime"
	"strconv"
)

var (
	// unused variables (in this implementation)
	// modified during test to exercise code paths in the cgo implementation.
	userBuffer  = 0
	groupBuffer = 0
)

func current() (*User, error) {
	uid := currentUID()
	// $USER and /etc/passwd may disagree; prefer the latter if we can get it.
	// See issue 27524 for more information.
	u, err := lookupUserId(uid)
	if err == nil {
		return u, nil
	}

	homeDir, _ := os.UserHomeDir()
	u = &User{
		Uid:      uid,
		Gid:      currentGID(),
		Username: os.Getenv("USER"),
		Name:     "", // ignored
		HomeDir:  homeDir,
	}
	// On Android, return a dummy user instead of failing.
	switch runtime.GOOS {
	case "android":
		if u.Uid == "" {
			u.Uid = "1"
		}
		if u.Username == "" {
			u.Username = "android"
		}
	}
	// cgo isn't available, but if we found the minimum information
	// without it, use it:
	if u.Uid != "" && u.Username != "" && u.HomeDir != "" {
		return u, nil
	}
	var missing string
	if u.Username == "" {
		missing = "$USER"
	}
	if u.HomeDir == "" {
		if missing != "" {
			missing += ", "
		}
		missing += "$HOME"
	}
	return u, fmt.Errorf("user: Current requires cgo or %s set in environment", missing)
}

func currentUID() string {
	if id := os.Getuid(); id >= 0 {
		return strconv.Itoa(id)
	}
	// Note: Windows returns -1, but this file isn't used on
	// Windows anyway, so this empty return path shouldn't be
	// used.
	return ""
}

func currentGID() string {
	if id := os.Getgid(); id >= 0 {
		return strconv.Itoa(id)
	}
	return ""
}
