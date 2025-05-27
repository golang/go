// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package cgrouptest provides best-effort helpers for running tests inside a
// cgroup.
package cgrouptest

import (
	"fmt"
	"internal/runtime/cgroup"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"syscall"
	"testing"
)

type CgroupV2 struct {
	orig string
	path string
}

func (c *CgroupV2) Path() string {
	return c.path
}

// Path to cpu.max.
func (c *CgroupV2) CPUMaxPath() string {
	return filepath.Join(c.path, "cpu.max")
}

// Set cpu.max. Pass -1 for quota to disable the limit.
func (c *CgroupV2) SetCPUMax(quota, period int64) error {
	q := "max"
	if quota >= 0 {
		q = strconv.FormatInt(quota, 10)
	}
	buf := fmt.Sprintf("%s %d", q, period)
	return os.WriteFile(c.CPUMaxPath(), []byte(buf), 0)
}

// InCgroupV2 creates a new v2 cgroup, migrates the current process into it,
// and then calls fn. When fn returns, the current process is migrated back to
// the original cgroup and the new cgroup is destroyed.
//
// If a new cgroup cannot be created, the test is skipped.
//
// This must not be used in parallel tests, as it affects the entire process.
func InCgroupV2(t *testing.T, fn func(*CgroupV2)) {
	mount, rel := findCurrent(t)
	parent := findOwnedParent(t, mount, rel)
	orig := filepath.Join(mount, rel)

	// Make sure the parent allows children to control cpu.
	b, err := os.ReadFile(filepath.Join(parent, "cgroup.subtree_control"))
	if err != nil {
		t.Skipf("unable to read cgroup.subtree_control: %v", err)
	}
	if !slices.Contains(strings.Fields(string(b)), "cpu") {
		// N.B. We should have permission to add cpu to
		// subtree_control, but it seems like a bad idea to change this
		// on a high-level cgroup that probably has lots of existing
		// children.
		t.Skipf("Parent cgroup %s does not allow children to control cpu, only %q", parent, string(b))
	}

	path, err := os.MkdirTemp(parent, "go-cgrouptest")
	if err != nil {
		t.Skipf("unable to create cgroup directory: %v", err)
	}
	// Important: defer cleanups so they run even in the event of panic.
	//
	// TODO(prattmic): Consider running everything in a subprocess just so
	// we can clean up if it throws or otherwise doesn't run the defers.
	defer func() {
		if err := os.Remove(path); err != nil {
			// Not much we can do, but at least inform of the
			// problem.
			t.Errorf("Error removing cgroup directory: %v", err)
		}
	}()

	migrateTo(t, path)
	defer migrateTo(t, orig)

	c := &CgroupV2{
		orig: orig,
		path: path,
	}
	fn(c)
}

// Returns the mount and relative directory of the current cgroup the process
// is in.
func findCurrent(t *testing.T) (string, string) {
	// Find the path to our current CPU cgroup. Currently this package is
	// only used for CPU cgroup testing, so the distinction of different
	// controllers doesn't matter.
	var scratch [cgroup.ParseSize]byte
	buf := make([]byte, cgroup.PathSize)
	n, err := cgroup.FindCPUMountPoint(buf, scratch[:])
	if err != nil {
		t.Skipf("cgroup: unable to find current cgroup mount: %v", err)
	}
	mount := string(buf[:n])

	n, ver, err := cgroup.FindCPURelativePath(buf, scratch[:])
	if err != nil {
		t.Skipf("cgroup: unable to find current cgroup path: %v", err)
	}
	if ver != cgroup.V2 {
		t.Skipf("cgroup: running on cgroup v%d want v2", ver)
	}
	rel := string(buf[1:n]) // The returned path always starts with /, skip it.
	rel = filepath.Join(".", rel) // Make sure this isn't empty string at root.
	return mount, rel
}

// Returns a parent directory in which we can create our own cgroup subdirectory.
func findOwnedParent(t *testing.T, mount, rel string) string {
	// There are many ways cgroups may be set up on a system. We don't try
	// to cover all of them, just common ones.
	//
	// To start with, systemd:
	//
	// Our test process is likely running inside a user session, in which
	// case we are likely inside a cgroup that looks something like:
	//
	//   /sys/fs/cgroup/user.slice/user-1234.slice/user@1234.service/vte-spawn-1.scope/
	//
	// Possibly with additional slice layers between user@1234.service and
	// the leaf scope.
	//
	// On new enough kernel and systemd versions (exact versions unknown),
	// full unprivileged control of the user's cgroups is permitted
	// directly via the cgroup filesystem. Specifically, the
	// user@1234.service directory is owned by the user, as are all
	// subdirectories.

	// We want to create our own subdirectory that we can migrate into and
	// then manipulate at will. It is tempting to create a new subdirectory
	// inside the current cgroup we are already in, however that will likey
	// not work. cgroup v2 only allows processes to be in leaf cgroups. Our
	// current cgroup likely contains multiple processes (at least this one
	// and the cmd/go test runner). If we make a subdirectory and try to
	// move our process into that cgroup, then the subdirectory and parent
	// would both contain processes. Linux won't allow us to do that [1].
	//
	// Instead, we will simply walk up to the highest directory that our
	// user owns and create our new subdirectory. Since that directory
	// already has a bunch of subdirectories, it must not directly contain
	// and processes.
	//
	// (This would fall apart if we already in the highest directory we
	// own, such as if there was simply a single cgroup for the entire
	// user. Luckily systemd at least does not do this.)
	//
	// [1] Minor technicality: By default a new subdirectory has no cgroup
	// controller (they must be explicitly enabled in the parent's
	// cgroup.subtree_control). Linux will allow moving processes into a
	// subdirectory that has no controllers while there are still processes
	// in the parent, but it won't allow adding controller until the parent
	// is empty. As far as I tell, the only purpose of this is to allow
	// reorganizing processes into a new set of subdirectories and then
	// adding controllers once done.
	root, err := os.OpenRoot(mount)
	if err != nil {
		t.Fatalf("error opening cgroup mount root: %v", err)
	}

	uid := os.Getuid()
	var prev string
	for rel != "." {
		fi, err := root.Stat(rel)
		if err != nil {
			t.Fatalf("error stating cgroup path: %v", err)
		}

		st := fi.Sys().(*syscall.Stat_t)
		if int(st.Uid) != uid {
			// Stop at first directory we don't own.
			break
		}

		prev = rel
		rel = filepath.Join(rel, "..")
	}

	if prev == "" {
		t.Skipf("No parent cgroup owned by UID %d", uid)
	}

	// We actually want the last directory where we were the owner.
	return filepath.Join(mount, prev)
}

// Migrate the current process to the cgroup directory dst.
func migrateTo(t *testing.T, dst string) {
	pid := []byte(strconv.FormatInt(int64(os.Getpid()), 10))
	if err := os.WriteFile(filepath.Join(dst, "cgroup.procs"), pid, 0); err != nil {
		t.Skipf("Unable to migrate into %s: %v", dst, err)
	}
}
