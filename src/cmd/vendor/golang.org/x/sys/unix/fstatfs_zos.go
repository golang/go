// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build zos && s390x
// +build zos,s390x

package unix

import (
	"unsafe"
)

// This file simulates fstatfs on z/OS using fstatvfs and w_getmntent.

func Fstatfs(fd int, stat *Statfs_t) (err error) {
	var stat_v Statvfs_t
	err = Fstatvfs(fd, &stat_v)
	if err == nil {
		// populate stat
		stat.Type = 0
		stat.Bsize = stat_v.Bsize
		stat.Blocks = stat_v.Blocks
		stat.Bfree = stat_v.Bfree
		stat.Bavail = stat_v.Bavail
		stat.Files = stat_v.Files
		stat.Ffree = stat_v.Ffree
		stat.Fsid = stat_v.Fsid
		stat.Namelen = stat_v.Namemax
		stat.Frsize = stat_v.Frsize
		stat.Flags = stat_v.Flag
		for passn := 0; passn < 5; passn++ {
			switch passn {
			case 0:
				err = tryGetmntent64(stat)
				break
			case 1:
				err = tryGetmntent128(stat)
				break
			case 2:
				err = tryGetmntent256(stat)
				break
			case 3:
				err = tryGetmntent512(stat)
				break
			case 4:
				err = tryGetmntent1024(stat)
				break
			default:
				break
			}
			//proceed to return if: err is nil (found), err is nonnil but not ERANGE (another error occurred)
			if err == nil || err != nil && err != ERANGE {
				break
			}
		}
	}
	return err
}

func tryGetmntent64(stat *Statfs_t) (err error) {
	var mnt_ent_buffer struct {
		header       W_Mnth
		filesys_info [64]W_Mntent
	}
	var buffer_size int = int(unsafe.Sizeof(mnt_ent_buffer))
	fs_count, err := W_Getmntent((*byte)(unsafe.Pointer(&mnt_ent_buffer)), buffer_size)
	if err != nil {
		return err
	}
	err = ERANGE //return ERANGE if no match is found in this batch
	for i := 0; i < fs_count; i++ {
		if stat.Fsid == uint64(mnt_ent_buffer.filesys_info[i].Dev) {
			stat.Type = uint32(mnt_ent_buffer.filesys_info[i].Fstname[0])
			err = nil
			break
		}
	}
	return err
}

func tryGetmntent128(stat *Statfs_t) (err error) {
	var mnt_ent_buffer struct {
		header       W_Mnth
		filesys_info [128]W_Mntent
	}
	var buffer_size int = int(unsafe.Sizeof(mnt_ent_buffer))
	fs_count, err := W_Getmntent((*byte)(unsafe.Pointer(&mnt_ent_buffer)), buffer_size)
	if err != nil {
		return err
	}
	err = ERANGE //return ERANGE if no match is found in this batch
	for i := 0; i < fs_count; i++ {
		if stat.Fsid == uint64(mnt_ent_buffer.filesys_info[i].Dev) {
			stat.Type = uint32(mnt_ent_buffer.filesys_info[i].Fstname[0])
			err = nil
			break
		}
	}
	return err
}

func tryGetmntent256(stat *Statfs_t) (err error) {
	var mnt_ent_buffer struct {
		header       W_Mnth
		filesys_info [256]W_Mntent
	}
	var buffer_size int = int(unsafe.Sizeof(mnt_ent_buffer))
	fs_count, err := W_Getmntent((*byte)(unsafe.Pointer(&mnt_ent_buffer)), buffer_size)
	if err != nil {
		return err
	}
	err = ERANGE //return ERANGE if no match is found in this batch
	for i := 0; i < fs_count; i++ {
		if stat.Fsid == uint64(mnt_ent_buffer.filesys_info[i].Dev) {
			stat.Type = uint32(mnt_ent_buffer.filesys_info[i].Fstname[0])
			err = nil
			break
		}
	}
	return err
}

func tryGetmntent512(stat *Statfs_t) (err error) {
	var mnt_ent_buffer struct {
		header       W_Mnth
		filesys_info [512]W_Mntent
	}
	var buffer_size int = int(unsafe.Sizeof(mnt_ent_buffer))
	fs_count, err := W_Getmntent((*byte)(unsafe.Pointer(&mnt_ent_buffer)), buffer_size)
	if err != nil {
		return err
	}
	err = ERANGE //return ERANGE if no match is found in this batch
	for i := 0; i < fs_count; i++ {
		if stat.Fsid == uint64(mnt_ent_buffer.filesys_info[i].Dev) {
			stat.Type = uint32(mnt_ent_buffer.filesys_info[i].Fstname[0])
			err = nil
			break
		}
	}
	return err
}

func tryGetmntent1024(stat *Statfs_t) (err error) {
	var mnt_ent_buffer struct {
		header       W_Mnth
		filesys_info [1024]W_Mntent
	}
	var buffer_size int = int(unsafe.Sizeof(mnt_ent_buffer))
	fs_count, err := W_Getmntent((*byte)(unsafe.Pointer(&mnt_ent_buffer)), buffer_size)
	if err != nil {
		return err
	}
	err = ERANGE //return ERANGE if no match is found in this batch
	for i := 0; i < fs_count; i++ {
		if stat.Fsid == uint64(mnt_ent_buffer.filesys_info[i].Dev) {
			stat.Type = uint32(mnt_ent_buffer.filesys_info[i].Fstname[0])
			err = nil
			break
		}
	}
	return err
}
