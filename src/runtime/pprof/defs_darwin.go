// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file is used as input to cgo --godefs (GOOS=arm64 or amd64) to
// generate the types used in viminfo_darwin_{arm64,amd64}.go which are
// hand edited as appropriate, primarily to avoid exporting the types.

//go:build ignore

package pprof

/*
#include <sys/param.h>
#include <mach/vm_prot.h>
#include <mach/vm_region.h>
*/
import "C"

type machVMRegionBasicInfoData C.vm_region_basic_info_data_64_t

const (
	_VM_PROT_READ    = C.VM_PROT_READ
	_VM_PROT_WRITE   = C.VM_PROT_WRITE
	_VM_PROT_EXECUTE = C.VM_PROT_EXECUTE

	_MACH_SEND_INVALID_DEST = C.MACH_SEND_INVALID_DEST

	_MAXPATHLEN = C.MAXPATHLEN
)
