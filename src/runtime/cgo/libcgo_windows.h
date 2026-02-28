// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Ensure there's one symbol marked __declspec(dllexport).
// If there are no exported symbols, the unfortunate behavior of
// the binutils linker is to also strip the relocations table,
// resulting in non-PIE binary. The other option is the
// --export-all-symbols flag, but we don't need to export all symbols
// and this may overflow the export table (#40795).
// See https://sourceware.org/bugzilla/show_bug.cgi?id=19011
__declspec(dllexport) int _cgo_dummy_export;
