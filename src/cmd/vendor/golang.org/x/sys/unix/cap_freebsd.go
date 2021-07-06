// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build freebsd
// +build freebsd

package unix

import (
	"errors"
	"fmt"
)

// Go implementation of C mostly found in /usr/src/sys/kern/subr_capability.c

const (
	// This is the version of CapRights this package understands. See C implementation for parallels.
	capRightsGoVersion = CAP_RIGHTS_VERSION_00
	capArSizeMin       = CAP_RIGHTS_VERSION_00 + 2
	capArSizeMax       = capRightsGoVersion + 2
)

var (
	bit2idx = []int{
		-1, 0, 1, -1, 2, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1,
		4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	}
)

func capidxbit(right uint64) int {
	return int((right >> 57) & 0x1f)
}

func rightToIndex(right uint64) (int, error) {
	idx := capidxbit(right)
	if idx < 0 || idx >= len(bit2idx) {
		return -2, fmt.Errorf("index for right 0x%x out of range", right)
	}
	return bit2idx[idx], nil
}

func caprver(right uint64) int {
	return int(right >> 62)
}

func capver(rights *CapRights) int {
	return caprver(rights.Rights[0])
}

func caparsize(rights *CapRights) int {
	return capver(rights) + 2
}

// CapRightsSet sets the permissions in setrights in rights.
func CapRightsSet(rights *CapRights, setrights []uint64) error {
	// This is essentially a copy of cap_rights_vset()
	if capver(rights) != CAP_RIGHTS_VERSION_00 {
		return fmt.Errorf("bad rights version %d", capver(rights))
	}

	n := caparsize(rights)
	if n < capArSizeMin || n > capArSizeMax {
		return errors.New("bad rights size")
	}

	for _, right := range setrights {
		if caprver(right) != CAP_RIGHTS_VERSION_00 {
			return errors.New("bad right version")
		}
		i, err := rightToIndex(right)
		if err != nil {
			return err
		}
		if i >= n {
			return errors.New("index overflow")
		}
		if capidxbit(rights.Rights[i]) != capidxbit(right) {
			return errors.New("index mismatch")
		}
		rights.Rights[i] |= right
		if capidxbit(rights.Rights[i]) != capidxbit(right) {
			return errors.New("index mismatch (after assign)")
		}
	}

	return nil
}

// CapRightsClear clears the permissions in clearrights from rights.
func CapRightsClear(rights *CapRights, clearrights []uint64) error {
	// This is essentially a copy of cap_rights_vclear()
	if capver(rights) != CAP_RIGHTS_VERSION_00 {
		return fmt.Errorf("bad rights version %d", capver(rights))
	}

	n := caparsize(rights)
	if n < capArSizeMin || n > capArSizeMax {
		return errors.New("bad rights size")
	}

	for _, right := range clearrights {
		if caprver(right) != CAP_RIGHTS_VERSION_00 {
			return errors.New("bad right version")
		}
		i, err := rightToIndex(right)
		if err != nil {
			return err
		}
		if i >= n {
			return errors.New("index overflow")
		}
		if capidxbit(rights.Rights[i]) != capidxbit(right) {
			return errors.New("index mismatch")
		}
		rights.Rights[i] &= ^(right & 0x01FFFFFFFFFFFFFF)
		if capidxbit(rights.Rights[i]) != capidxbit(right) {
			return errors.New("index mismatch (after assign)")
		}
	}

	return nil
}

// CapRightsIsSet checks whether all the permissions in setrights are present in rights.
func CapRightsIsSet(rights *CapRights, setrights []uint64) (bool, error) {
	// This is essentially a copy of cap_rights_is_vset()
	if capver(rights) != CAP_RIGHTS_VERSION_00 {
		return false, fmt.Errorf("bad rights version %d", capver(rights))
	}

	n := caparsize(rights)
	if n < capArSizeMin || n > capArSizeMax {
		return false, errors.New("bad rights size")
	}

	for _, right := range setrights {
		if caprver(right) != CAP_RIGHTS_VERSION_00 {
			return false, errors.New("bad right version")
		}
		i, err := rightToIndex(right)
		if err != nil {
			return false, err
		}
		if i >= n {
			return false, errors.New("index overflow")
		}
		if capidxbit(rights.Rights[i]) != capidxbit(right) {
			return false, errors.New("index mismatch")
		}
		if (rights.Rights[i] & right) != right {
			return false, nil
		}
	}

	return true, nil
}

func capright(idx uint64, bit uint64) uint64 {
	return ((1 << (57 + idx)) | bit)
}

// CapRightsInit returns a pointer to an initialised CapRights structure filled with rights.
// See man cap_rights_init(3) and rights(4).
func CapRightsInit(rights []uint64) (*CapRights, error) {
	var r CapRights
	r.Rights[0] = (capRightsGoVersion << 62) | capright(0, 0)
	r.Rights[1] = capright(1, 0)

	err := CapRightsSet(&r, rights)
	if err != nil {
		return nil, err
	}
	return &r, nil
}

// CapRightsLimit reduces the operations permitted on fd to at most those contained in rights.
// The capability rights on fd can never be increased by CapRightsLimit.
// See man cap_rights_limit(2) and rights(4).
func CapRightsLimit(fd uintptr, rights *CapRights) error {
	return capRightsLimit(int(fd), rights)
}

// CapRightsGet returns a CapRights structure containing the operations permitted on fd.
// See man cap_rights_get(3) and rights(4).
func CapRightsGet(fd uintptr) (*CapRights, error) {
	r, err := CapRightsInit(nil)
	if err != nil {
		return nil, err
	}
	err = capRightsGet(capRightsGoVersion, int(fd), r)
	if err != nil {
		return nil, err
	}
	return r, nil
}
