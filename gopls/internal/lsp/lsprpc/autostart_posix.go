// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris
// +build darwin dragonfly freebsd linux netbsd openbsd solaris

package lsprpc

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"log"
	"os"
	"os/user"
	"path/filepath"
	"strconv"
	"syscall"

	exec "golang.org/x/sys/execabs"
)

func init() {
	daemonize = daemonizePosix
	autoNetworkAddress = autoNetworkAddressPosix
	verifyRemoteOwnership = verifyRemoteOwnershipPosix
}

func daemonizePosix(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Setsid: true,
	}
}

// autoNetworkAddressPosix resolves an id on the 'auto' pseduo-network to a
// real network and address. On unix, this uses unix domain sockets.
func autoNetworkAddressPosix(goplsPath, id string) (network string, address string) {
	// Especially when doing local development or testing, it's important that
	// the remote gopls instance we connect to is running the same binary as our
	// forwarder. So we encode a short hash of the binary path into the daemon
	// socket name. If possible, we also include the buildid in this hash, to
	// account for long-running processes where the binary has been subsequently
	// rebuilt.
	h := sha256.New()
	cmd := exec.Command("go", "tool", "buildid", goplsPath)
	cmd.Stdout = h
	var pathHash []byte
	if err := cmd.Run(); err == nil {
		pathHash = h.Sum(nil)
	} else {
		log.Printf("error getting current buildid: %v", err)
		sum := sha256.Sum256([]byte(goplsPath))
		pathHash = sum[:]
	}
	shortHash := fmt.Sprintf("%x", pathHash)[:6]
	user := os.Getenv("USER")
	if user == "" {
		user = "shared"
	}
	basename := filepath.Base(goplsPath)
	idComponent := ""
	if id != "" {
		idComponent = "-" + id
	}
	runtimeDir := os.TempDir()
	if xdg := os.Getenv("XDG_RUNTIME_DIR"); xdg != "" {
		runtimeDir = xdg
	}
	return "unix", filepath.Join(runtimeDir, fmt.Sprintf("%s-%s-daemon.%s%s", basename, shortHash, user, idComponent))
}

func verifyRemoteOwnershipPosix(network, address string) (bool, error) {
	if network != "unix" {
		return true, nil
	}
	fi, err := os.Stat(address)
	if err != nil {
		if os.IsNotExist(err) {
			return true, nil
		}
		return false, fmt.Errorf("checking socket owner: %w", err)
	}
	stat, ok := fi.Sys().(*syscall.Stat_t)
	if !ok {
		return false, errors.New("fi.Sys() is not a Stat_t")
	}
	user, err := user.Current()
	if err != nil {
		return false, fmt.Errorf("checking current user: %w", err)
	}
	uid, err := strconv.ParseUint(user.Uid, 10, 32)
	if err != nil {
		return false, fmt.Errorf("parsing current UID: %w", err)
	}
	return stat.Uid == uint32(uid), nil
}
