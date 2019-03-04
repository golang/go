// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin

package net

import (
	"context"
	"errors"
	"unsafe"

	"github.com/golang/go.bkp/src/runtime"
)

func resSearch(ctx context.Context, hostname string) ([]IPAddr, error) {

	var byteHostname = []byte(hostname)
	var responseBuffer = [512]byte{}
	retcode := runtime.Res_search(&byteHostname[0], 1, 1, &responseBuffer[0], 512)
	if retcode < 0 {
		return nil, errors.New("//TODO:")
	}

}

//go:nosplit
//go:cgo_unsafe_args
func res_search(name *byte, class int32, rtype int32, answer *byte, anslen int32) int32 {
	return libcCall(unsafe.Pointer(funcPC(res_search_trampoline)), unsafe.Pointer(&name))
}
func res_search_trampoline()

//go:cgo_import_dynamic libc_res_search res_search "/usr/lib/libSystem.B.dylib"
