// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * Register 'f' symbol file fragments.  Doing this while parsing the
 * .6 input saves a pass over the symbol table later.
 */
void dwarfaddfrag(int n, char* frag);

/*
 * Emit debug_abbrevs, debug_info and debug_line sections to current
 * offset in cout.
 */
void dwarfemitdebugsections(void);

/*
 * Add the dwarf section names to the ELF
 * s[ection]h[eader]str[ing]tab.  Prerequisite for
 * dwarfaddelfheaders().
 */
void dwarfaddshstrings(Sym *shstrtab);

/*
 * Add section headers pointing to the sections emitted in
 * dwarfemitdebugsections.
 */
void dwarfaddelfheaders(void);
void dwarfaddmachoheaders(void);
void dwarfaddpeheaders(void);
