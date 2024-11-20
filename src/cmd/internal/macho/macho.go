// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package macho provides functionalities to handle Mach-O
// beyond the debug/macho package, for the toolchain.
package macho

import (
	"debug/macho"
	"encoding/binary"
	"io"
	"unsafe"
)

const (
	LC_SEGMENT                  = 0x1
	LC_SYMTAB                   = 0x2
	LC_SYMSEG                   = 0x3
	LC_THREAD                   = 0x4
	LC_UNIXTHREAD               = 0x5
	LC_LOADFVMLIB               = 0x6
	LC_IDFVMLIB                 = 0x7
	LC_IDENT                    = 0x8
	LC_FVMFILE                  = 0x9
	LC_PREPAGE                  = 0xa
	LC_DYSYMTAB                 = 0xb
	LC_LOAD_DYLIB               = 0xc
	LC_ID_DYLIB                 = 0xd
	LC_LOAD_DYLINKER            = 0xe
	LC_ID_DYLINKER              = 0xf
	LC_PREBOUND_DYLIB           = 0x10
	LC_ROUTINES                 = 0x11
	LC_SUB_FRAMEWORK            = 0x12
	LC_SUB_UMBRELLA             = 0x13
	LC_SUB_CLIENT               = 0x14
	LC_SUB_LIBRARY              = 0x15
	LC_TWOLEVEL_HINTS           = 0x16
	LC_PREBIND_CKSUM            = 0x17
	LC_LOAD_WEAK_DYLIB          = 0x80000018
	LC_SEGMENT_64               = 0x19
	LC_ROUTINES_64              = 0x1a
	LC_UUID                     = 0x1b
	LC_RPATH                    = 0x8000001c
	LC_CODE_SIGNATURE           = 0x1d
	LC_SEGMENT_SPLIT_INFO       = 0x1e
	LC_REEXPORT_DYLIB           = 0x8000001f
	LC_LAZY_LOAD_DYLIB          = 0x20
	LC_ENCRYPTION_INFO          = 0x21
	LC_DYLD_INFO                = 0x22
	LC_DYLD_INFO_ONLY           = 0x80000022
	LC_LOAD_UPWARD_DYLIB        = 0x80000023
	LC_VERSION_MIN_MACOSX       = 0x24
	LC_VERSION_MIN_IPHONEOS     = 0x25
	LC_FUNCTION_STARTS          = 0x26
	LC_DYLD_ENVIRONMENT         = 0x27
	LC_MAIN                     = 0x80000028
	LC_DATA_IN_CODE             = 0x29
	LC_SOURCE_VERSION           = 0x2A
	LC_DYLIB_CODE_SIGN_DRS      = 0x2B
	LC_ENCRYPTION_INFO_64       = 0x2C
	LC_LINKER_OPTION            = 0x2D
	LC_LINKER_OPTIMIZATION_HINT = 0x2E
	LC_VERSION_MIN_TVOS         = 0x2F
	LC_VERSION_MIN_WATCHOS      = 0x30
	LC_VERSION_NOTE             = 0x31
	LC_BUILD_VERSION            = 0x32
	LC_DYLD_EXPORTS_TRIE        = 0x80000033
	LC_DYLD_CHAINED_FIXUPS      = 0x80000034
)

// LoadCmd is macho.LoadCmd with its length, which is also
// the load command header in the Mach-O file.
type LoadCmd struct {
	Cmd macho.LoadCmd
	Len uint32
}

type LoadCmdReader struct {
	offset, next int64
	f            io.ReadSeeker
	order        binary.ByteOrder
}

func NewLoadCmdReader(f io.ReadSeeker, order binary.ByteOrder, nextOffset int64) LoadCmdReader {
	return LoadCmdReader{next: nextOffset, f: f, order: order}
}

func (r *LoadCmdReader) Next() (LoadCmd, error) {
	var cmd LoadCmd

	r.offset = r.next
	if _, err := r.f.Seek(r.offset, 0); err != nil {
		return cmd, err
	}
	if err := binary.Read(r.f, r.order, &cmd); err != nil {
		return cmd, err
	}
	r.next = r.offset + int64(cmd.Len)
	return cmd, nil
}

func (r LoadCmdReader) ReadAt(offset int64, data interface{}) error {
	if _, err := r.f.Seek(r.offset+offset, 0); err != nil {
		return err
	}
	return binary.Read(r.f, r.order, data)
}

func (r LoadCmdReader) Offset() int64 { return r.offset }

type LoadCmdUpdater struct {
	LoadCmdReader
}

func NewLoadCmdUpdater(f io.ReadWriteSeeker, order binary.ByteOrder, nextOffset int64) LoadCmdUpdater {
	return LoadCmdUpdater{NewLoadCmdReader(f, order, nextOffset)}
}

func (u LoadCmdUpdater) WriteAt(offset int64, data interface{}) error {
	if _, err := u.f.Seek(u.offset+offset, 0); err != nil {
		return err
	}
	return binary.Write(u.f.(io.Writer), u.order, data)
}

func FileHeaderSize(f *macho.File) int64 {
	offset := int64(unsafe.Sizeof(f.FileHeader))
	if is64bit := f.Magic == macho.Magic64; is64bit {
		// mach_header_64 has one extra uint32.
		offset += 4
	}
	return offset
}
