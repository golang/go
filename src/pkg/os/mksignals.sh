#!/bin/sh

for targ in \
	darwin_386 \
	darwin_amd64 \
	freebsd_386 \
	freebsd_amd64 \
	linux_386 \
	linux_amd64 \
	linux_arm \
	openbsd_386 \
	openbsd_amd64 \
; do
	./mkunixsignals.sh ../syscall/zerrors_$targ.go |gofmt >zsignal_$targ.go
done

for targ in \
	windows_386 \
	windows_amd64 \
; do
	./mkunixsignals.sh ../syscall/ztypes_windows.go |gofmt >zsignal_$targ.go
done
