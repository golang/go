// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package seccomp

/*
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <errno.h>
#include <stddef.h>
#include <unistd.h>
#include <stdint.h>

// A few definitions copied from linux/filter.h and linux/seccomp.h,
// which might not be available on all systems.

struct sock_filter {
    uint16_t code;
    uint8_t jt;
    uint8_t jf;
    uint32_t k;
};

struct sock_fprog {
    unsigned short len;
    struct sock_filter *filter;
};

#define BPF_LD	0x00
#define BPF_W	0x00
#define BPF_ABS	0x20
#define BPF_JMP	0x05
#define BPF_JEQ	0x10
#define BPF_K	0x00
#define BPF_RET	0x06

#define BPF_STMT(code, k) { (unsigned short)(code), 0, 0, k }
#define BPF_JUMP(code, k, jt, jf) { (unsigned short)(code), jt, jf, k }

struct seccomp_data {
	int nr;
	uint32_t arch;
	uint64_t instruction_pointer;
	uint64_t args[6];
};

#define SECCOMP_RET_ERRNO 0x00050000U
#define SECCOMP_RET_ALLOW 0x7fff0000U
#define SECCOMP_SET_MODE_FILTER 1

#ifndef SYS_getrandom
#define SYS_getrandom -1
#endif

#ifndef SYS_seccomp
#define SYS_seccomp -1
#endif

int disable_getrandom() {
    if (SYS_getrandom == -1 || SYS_seccomp == -1) {
        return 3;
    }
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)) {
        return 1;
    }
    struct sock_filter filter[] = {
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, (offsetof(struct seccomp_data, nr))),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_getrandom, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | ENOSYS),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
    };
    struct sock_fprog prog = {
        .len = sizeof(filter) / sizeof((filter)[0]),
        .filter = filter,
    };
    if (syscall(SYS_seccomp, SECCOMP_SET_MODE_FILTER, 0, &prog)) {
        return 2;
    }
    return 0;
}
*/
import "C"
import "fmt"

// DisableGetrandom makes future calls to getrandom(2) fail with ENOSYS. It
// applies only to the current thread and to any programs executed from it.
// Callers should use [runtime.LockOSThread] in a dedicated goroutine.
func DisableGetrandom() error {
	if errno := C.disable_getrandom(); errno != 0 {
		return fmt.Errorf("failed to disable getrandom: %v", errno)
	}
	return nil
}
