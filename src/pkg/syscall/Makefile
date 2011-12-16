# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.inc

TARG=syscall
GOFILES=\
	str.go\
	syscall.go\
	syscall_$(GOARCH).go\
	syscall_$(GOOS).go\
	syscall_$(GOOS)_$(GOARCH).go\
	zerrors_$(GOOS)_$(GOARCH).go\
	zsyscall_$(GOOS)_$(GOARCH).go\
	zsysnum_$(GOOS)_$(GOARCH).go\
	ztypes_$(GOOS)_$(GOARCH).go\

GOFILES_darwin=\
	bpf_bsd.go\
	env_unix.go\
	exec_unix.go\
	route_bsd.go\
	route_darwin.go\
	sockcmsg_unix.go\
	syscall_bsd.go\
	syscall_unix.go\

GOFILES_freebsd=\
	bpf_bsd.go\
	env_unix.go\
	exec_unix.go\
	route_bsd.go\
	route_freebsd.go\
	sockcmsg_unix.go\
	syscall_bsd.go\
	syscall_unix.go\

GOFILES_linux=\
	env_unix.go\
	exec_unix.go\
	lsf_linux.go\
	netlink_linux.go\
	sockcmsg_linux.go\
	sockcmsg_unix.go\
	syscall_unix.go\

GOFILES_netbsd=\
	bpf_bsd.go\
	env_unix.go\
	exec_unix.go\
	route_bsd.go\
	route_netbsd.go\
	sockcmsg_unix.go\
	syscall_bsd.go\
	syscall_unix.go\
	zsysctl_netbsd.go\

GOFILES_openbsd=\
	bpf_bsd.go\
	env_unix.go\
	exec_unix.go\
	route_bsd.go\
	route_openbsd.go\
	sockcmsg_unix.go\
	syscall_bsd.go\
	syscall_unix.go\
	zsysctl_openbsd.go\

GOFILES_plan9=\
	env_plan9.go\
	exec_plan9.go\

GOFILES_windows=\
	env_windows.go\
	exec_windows.go\
	dll_windows.go\
	zerrors_windows.go\
	ztypes_windows.go\

OFILES=\
	asm_$(GOOS)_$(GOARCH).$O\

GOFILES+=$(GOFILES_$(GOOS))

include ../../Make.pkg
