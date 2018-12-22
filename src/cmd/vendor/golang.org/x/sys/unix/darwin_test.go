// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin,go1.12,amd64 darwin,go1.12,386

package unix

import (
	"os"
	"os/exec"
	"strings"
	"testing"
)

type darwinTest struct {
	name string
	f    func()
}

// TODO(khr): decide whether to keep this test enabled permanently or
// only temporarily.
func TestDarwinLoader(t *testing.T) {
	// Make sure the Darwin dynamic loader can actually resolve
	// all the system calls into libSystem.dylib. Unfortunately
	// there is no easy way to test this at compile time. So we
	// implement a crazy hack here, calling into the syscall
	// function with all its arguments set to junk, and see what
	// error we get. We are happy with any error (or none) except
	// an error from the dynamic loader.
	//
	// We have to run each test in a separate subprocess for fault isolation.
	//
	// Hopefully the junk args won't accidentally ask the system to do "rm -fr /".
	//
	// In an ideal world each syscall would have its own test, so this test
	// would be unnecessary. Unfortunately, we do not live in that world.
	for _, test := range darwinTests {
		// Call the test binary recursively, giving it a magic argument
		// (see init below) and the name of the test to run.
		cmd := exec.Command(os.Args[0], "testDarwinLoader", test.name)

		// Run subprocess, collect results. Note that we expect the subprocess
		// to fail somehow, so the error is irrelevant.
		out, _ := cmd.CombinedOutput()

		if strings.Contains(string(out), "dyld: Symbol not found:") {
			t.Errorf("can't resolve %s in libSystem.dylib", test.name)
		}
		if !strings.Contains(string(out), "success") {
			// Not really an error. Might be a syscall that never returns,
			// like exit, or one that segfaults, like gettimeofday.
			t.Logf("test never finished: %s: %s", test.name, string(out))
		}
	}
}

func init() {
	// The test binary execs itself with the "testDarwinLoader" argument.
	// Run the test specified by os.Args[2], then panic.
	if len(os.Args) >= 3 && os.Args[1] == "testDarwinLoader" {
		for _, test := range darwinTests {
			if test.name == os.Args[2] {
				test.f()
			}
		}
		// Panic with a "success" label, so the parent process can check it.
		panic("success")
	}
}

// All the _trampoline functions in zsyscall_darwin_$ARCH.s
var darwinTests = [...]darwinTest{
	{"getgroups", libc_getgroups_trampoline},
	{"setgroups", libc_setgroups_trampoline},
	{"wait4", libc_wait4_trampoline},
	{"accept", libc_accept_trampoline},
	{"bind", libc_bind_trampoline},
	{"connect", libc_connect_trampoline},
	{"socket", libc_socket_trampoline},
	{"getsockopt", libc_getsockopt_trampoline},
	{"setsockopt", libc_setsockopt_trampoline},
	{"getpeername", libc_getpeername_trampoline},
	{"getsockname", libc_getsockname_trampoline},
	{"shutdown", libc_shutdown_trampoline},
	{"socketpair", libc_socketpair_trampoline},
	{"recvfrom", libc_recvfrom_trampoline},
	{"sendto", libc_sendto_trampoline},
	{"recvmsg", libc_recvmsg_trampoline},
	{"sendmsg", libc_sendmsg_trampoline},
	{"kevent", libc_kevent_trampoline},
	{"__sysctl", libc___sysctl_trampoline},
	{"utimes", libc_utimes_trampoline},
	{"futimes", libc_futimes_trampoline},
	{"fcntl", libc_fcntl_trampoline},
	{"poll", libc_poll_trampoline},
	{"madvise", libc_madvise_trampoline},
	{"mlock", libc_mlock_trampoline},
	{"mlockall", libc_mlockall_trampoline},
	{"mprotect", libc_mprotect_trampoline},
	{"msync", libc_msync_trampoline},
	{"munlock", libc_munlock_trampoline},
	{"munlockall", libc_munlockall_trampoline},
	{"ptrace", libc_ptrace_trampoline},
	{"pipe", libc_pipe_trampoline},
	{"getxattr", libc_getxattr_trampoline},
	{"fgetxattr", libc_fgetxattr_trampoline},
	{"setxattr", libc_setxattr_trampoline},
	{"fsetxattr", libc_fsetxattr_trampoline},
	{"removexattr", libc_removexattr_trampoline},
	{"fremovexattr", libc_fremovexattr_trampoline},
	{"listxattr", libc_listxattr_trampoline},
	{"flistxattr", libc_flistxattr_trampoline},
	{"kill", libc_kill_trampoline},
	{"ioctl", libc_ioctl_trampoline},
	{"access", libc_access_trampoline},
	{"adjtime", libc_adjtime_trampoline},
	{"chdir", libc_chdir_trampoline},
	{"chflags", libc_chflags_trampoline},
	{"chmod", libc_chmod_trampoline},
	{"chown", libc_chown_trampoline},
	{"chroot", libc_chroot_trampoline},
	{"close", libc_close_trampoline},
	{"dup", libc_dup_trampoline},
	{"dup2", libc_dup2_trampoline},
	{"exchangedata", libc_exchangedata_trampoline},
	{"exit", libc_exit_trampoline},
	{"faccessat", libc_faccessat_trampoline},
	{"fchdir", libc_fchdir_trampoline},
	{"fchflags", libc_fchflags_trampoline},
	{"fchmod", libc_fchmod_trampoline},
	{"fchmodat", libc_fchmodat_trampoline},
	{"fchown", libc_fchown_trampoline},
	{"fchownat", libc_fchownat_trampoline},
	{"flock", libc_flock_trampoline},
	{"fpathconf", libc_fpathconf_trampoline},
	{"fstat64", libc_fstat64_trampoline},
	{"fstatat64", libc_fstatat64_trampoline},
	{"fstatfs64", libc_fstatfs64_trampoline},
	{"fsync", libc_fsync_trampoline},
	{"ftruncate", libc_ftruncate_trampoline},
	{"__getdirentries64", libc___getdirentries64_trampoline},
	{"getdtablesize", libc_getdtablesize_trampoline},
	{"getegid", libc_getegid_trampoline},
	{"geteuid", libc_geteuid_trampoline},
	{"getgid", libc_getgid_trampoline},
	{"getpgid", libc_getpgid_trampoline},
	{"getpgrp", libc_getpgrp_trampoline},
	{"getpid", libc_getpid_trampoline},
	{"getppid", libc_getppid_trampoline},
	{"getpriority", libc_getpriority_trampoline},
	{"getrlimit", libc_getrlimit_trampoline},
	{"getrusage", libc_getrusage_trampoline},
	{"getsid", libc_getsid_trampoline},
	{"getuid", libc_getuid_trampoline},
	{"issetugid", libc_issetugid_trampoline},
	{"kqueue", libc_kqueue_trampoline},
	{"lchown", libc_lchown_trampoline},
	{"link", libc_link_trampoline},
	{"linkat", libc_linkat_trampoline},
	{"listen", libc_listen_trampoline},
	{"lstat64", libc_lstat64_trampoline},
	{"mkdir", libc_mkdir_trampoline},
	{"mkdirat", libc_mkdirat_trampoline},
	{"mkfifo", libc_mkfifo_trampoline},
	{"mknod", libc_mknod_trampoline},
	{"open", libc_open_trampoline},
	{"openat", libc_openat_trampoline},
	{"pathconf", libc_pathconf_trampoline},
	{"pread", libc_pread_trampoline},
	{"pwrite", libc_pwrite_trampoline},
	{"read", libc_read_trampoline},
	{"readlink", libc_readlink_trampoline},
	{"readlinkat", libc_readlinkat_trampoline},
	{"rename", libc_rename_trampoline},
	{"renameat", libc_renameat_trampoline},
	{"revoke", libc_revoke_trampoline},
	{"rmdir", libc_rmdir_trampoline},
	{"lseek", libc_lseek_trampoline},
	{"select", libc_select_trampoline},
	{"setegid", libc_setegid_trampoline},
	{"seteuid", libc_seteuid_trampoline},
	{"setgid", libc_setgid_trampoline},
	{"setlogin", libc_setlogin_trampoline},
	{"setpgid", libc_setpgid_trampoline},
	{"setpriority", libc_setpriority_trampoline},
	{"setprivexec", libc_setprivexec_trampoline},
	{"setregid", libc_setregid_trampoline},
	{"setreuid", libc_setreuid_trampoline},
	{"setrlimit", libc_setrlimit_trampoline},
	{"setsid", libc_setsid_trampoline},
	{"settimeofday", libc_settimeofday_trampoline},
	{"setuid", libc_setuid_trampoline},
	{"stat64", libc_stat64_trampoline},
	{"statfs64", libc_statfs64_trampoline},
	{"symlink", libc_symlink_trampoline},
	{"symlinkat", libc_symlinkat_trampoline},
	{"sync", libc_sync_trampoline},
	{"truncate", libc_truncate_trampoline},
	{"umask", libc_umask_trampoline},
	{"undelete", libc_undelete_trampoline},
	{"unlink", libc_unlink_trampoline},
	{"unlinkat", libc_unlinkat_trampoline},
	{"unmount", libc_unmount_trampoline},
	{"write", libc_write_trampoline},
	{"mmap", libc_mmap_trampoline},
	{"munmap", libc_munmap_trampoline},
	{"gettimeofday", libc_gettimeofday_trampoline},
	{"getfsstat64", libc_getfsstat64_trampoline},
}
