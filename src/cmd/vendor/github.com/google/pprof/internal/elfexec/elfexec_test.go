// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package elfexec

import (
	"debug/elf"
	"testing"
)

func TestGetBase(t *testing.T) {

	fhExec := &elf.FileHeader{
		Type: elf.ET_EXEC,
	}
	fhRel := &elf.FileHeader{
		Type: elf.ET_REL,
	}
	fhDyn := &elf.FileHeader{
		Type: elf.ET_DYN,
	}
	lsOffset := &elf.ProgHeader{
		Vaddr: 0x400000,
		Off:   0x200000,
	}
	kernelHeader := &elf.ProgHeader{
		Vaddr: 0xffffffff81000000,
	}
	ppc64KernelHeader := &elf.ProgHeader{
		Vaddr: 0xc000000000000000,
	}

	testcases := []struct {
		label                string
		fh                   *elf.FileHeader
		loadSegment          *elf.ProgHeader
		stextOffset          *uint64
		start, limit, offset uint64
		want                 uint64
		wanterr              bool
	}{
		{"exec", fhExec, nil, nil, 0x400000, 0, 0, 0, false},
		{"exec offset", fhExec, lsOffset, nil, 0x400000, 0x800000, 0, 0, false},
		{"exec offset 2", fhExec, lsOffset, nil, 0x200000, 0x600000, 0, 0, false},
		{"exec nomap", fhExec, nil, nil, 0, 0, 0, 0, false},
		{"exec kernel", fhExec, kernelHeader, uint64p(0xffffffff81000198), 0xffffffff82000198, 0xffffffff83000198, 0, 0x1000000, false},
		{"exec PPC64 kernel", fhExec, ppc64KernelHeader, uint64p(0xc000000000000000), 0xc000000000000000, 0xd00000001a730000, 0xc000000000000000, 0x0, false},
		{"exec chromeos kernel", fhExec, kernelHeader, uint64p(0xffffffff81000198), 0, 0x10197, 0, 0x7efffe68, false},
		{"exec chromeos kernel 2", fhExec, kernelHeader, uint64p(0xffffffff81000198), 0, 0x10198, 0, 0x7efffe68, false},
		{"exec chromeos kernel 3", fhExec, kernelHeader, uint64p(0xffffffff81000198), 0x198, 0x100000, 0, 0x7f000000, false},
		{"exec chromeos kernel 4", fhExec, kernelHeader, uint64p(0xffffffff81200198), 0x198, 0x100000, 0, 0x7ee00000, false},
		{"exec chromeos kernel unremapped", fhExec, kernelHeader, uint64p(0xffffffff810001c8), 0xffffffff834001c8, 0xffffffffc0000000, 0xffffffff834001c8, 0x2400000, false},
		{"dyn", fhDyn, nil, nil, 0x200000, 0x300000, 0, 0x200000, false},
		{"dyn offset", fhDyn, lsOffset, nil, 0x0, 0x300000, 0, 0xFFFFFFFFFFC00000, false},
		{"dyn nomap", fhDyn, nil, nil, 0x0, 0x0, 0, 0, false},
		{"rel", fhRel, nil, nil, 0x2000000, 0x3000000, 0, 0x2000000, false},
		{"rel nomap", fhRel, nil, nil, 0x0, ^uint64(0), 0, 0, false},
		{"rel offset", fhRel, nil, nil, 0x100000, 0x200000, 0x1, 0, true},
	}

	for _, tc := range testcases {
		base, err := GetBase(tc.fh, tc.loadSegment, tc.stextOffset, tc.start, tc.limit, tc.offset)
		if err != nil {
			if !tc.wanterr {
				t.Errorf("%s: want no error, got %v", tc.label, err)
			}
			continue
		}
		if tc.wanterr {
			t.Errorf("%s: want error, got nil", tc.label)
			continue
		}
		if base != tc.want {
			t.Errorf("%s: want %x, got %x", tc.label, tc.want, base)
		}
	}
}

func uint64p(n uint64) *uint64 {
	return &n
}
