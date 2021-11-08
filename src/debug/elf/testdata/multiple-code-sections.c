// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Build with:
// gcc -g multiple-code-sections.c -Wl,--emit-relocs -Wl,--discard-none -Wl,-zmax-page-size=1 -fno-asynchronous-unwind-tables -o go-relocation-test-gcc930-ranges-with-rela-x86-64
// gcc -g multiple-code-sections.c -Wl,-zmax-page-size=1 -fno-asynchronous-unwind-tables -o go-relocation-test-gcc930-ranges-no-rela-x86-64
// Strip with:
// strip --only-keep-debug \
//       --remove-section=.eh_frame \
//       --remove-section=.eh_frame_hdr \
//       --remove-section=.shstrtab \
//       --remove-section=.strtab \
//       --remove-section=.symtab \
//       --remove-section=.note.gnu.build-id \
//       --remove-section=.note.ABI-tag \
//       --remove-section=.dynamic \
//       --remove-section=.gnu.hash \
//       --remove-section=.interp \
//       --remove-section=.rodata
__attribute__((section(".separate_section"))) // To get GCC to emit a DW_AT_ranges attribute for the CU.
int func(void) {
    return 0;
}

int main(int argc, char *argv[]) {
    return 0;
}
