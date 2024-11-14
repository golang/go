// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
FIPS-140 Verification Support

# Overview

For FIPS-140 crypto certification, one of the requirements is that the
“cryptographic module” perform a power-on self-test that includes
verification of its code+data at startup, ostensibly to guard against
corruption. (Like most of FIPS, the actual value here is as questionable
as it is non-negotiable.) Specifically, at startup we need to compute
an HMAC-SHA256 of the cryptographic code+data and compare it against a
build-time HMAC-SHA256 that has been stored in the binary as well.
This obviously guards against accidental corruption only, not attacks.

We could compute an HMAC-SHA256 of the entire binary, but that's more
startup latency than we'd like. (At 500 MB/s, a large 50MB binary
would incur a 100ms hit.) Also, as we'll see, there are some
limitations imposed on the code+data being hashed, and it's nice to
restrict those to the actual cryptographic packages.

# FIPS Symbol Types

Since we're not hashing the whole binary, we need to record the parts
of the binary that contain FIPS code, specifically the part of the
binary corresponding to the crypto/internal/fips package subtree.
To do that, we create special symbol types STEXTFIPS, SRODATAFIPS,
SNOPTRDATAFIPS, and SDATAFIPS, which those packages use instead of
STEXT, SRODATA, SNOPTRDATA, and SDATA. The linker groups symbols by
their type, so that naturally makes the FIPS parts contiguous within a
given type. The linker then writes out in a special symbol the start
and end of each of these FIPS-specific sections, alongside the
expected HMAC-SHA256 of them. At startup, the crypto/internal/fips/check
package has an init function that recomputes the hash and checks it
against the recorded expectation.

The first important functionality in this file, then, is converting
from the standard symbol types to the FIPS symbol types, in the code
that needs them. Every time an LSym.Type is set, code must call
[LSym.setFIPSType] to update the Type to a FIPS type if appropriate.

# Relocation Restrictions

Of course, for the hashes to match, the FIPS code+data written by the
linker has to match the FIPS code+data in memory at init time.
This means that there cannot be an load-time relocations that modify
the FIPS code+data. In a standard -buildmode=exe build, that's vacuously
true, since those binaries have no load-time relocations at all.
For a -buildmode=pie build, there's more to be done.
Specifically, we have to make sure that all the relocations needed are
position-independent, so that they can be applied a link time with no
load-time component. For the code segment (the STEXTFIPS symbols),
that means only using PC-relative relocations. For the data segment,
that means basically having no relocations at all. In particular,
there cannot be R_ADDR relocations.

For example, consider the compilation of code like the global variables:

	var array = [...]int{10, 20, 30}
	var slice = array[:]

The standard implementation of these globals is to fill out the array
values in an SDATA symbol at link time, and then also to fill out the
slice header at link time as {nil, 3, 3}, along with a relocation to
fill in the first word of the slice header with the pointer &array at
load time, once the address of array is known.

A similar issue happens with:

	var slice = []int{10, 20, 30}

The compiler invents an anonymous array and then treats the code as in
the first example. In both cases, a load-time relocation applied
before the crypto/internal/fips/check init function would invalidate
the hash. Instead, we disable the “link time initialization” optimizations
in the compiler (package staticinit) for the fips packages.
That way, the slice initialization is deferred to its own init function.
As long as the package in question imports crypto/internal/fips/check,
the hash check will happen before the package's own init function
runs, and so the hash check will see the slice header written by the
linker, with a slice base pointer predictably nil instead of the
unpredictable &array address.

The details of disabling the static initialization appropriately are
left to the compiler (see ../../compile/internal/staticinit).
This file is only concerned with making sure that no hash-invalidating
relocations sneak into the object files. [LSym.checkFIPSReloc] is called
for every new relocation in a symbol in a FIPS package (as reported by
[Link.IsFIPS]) and rejects invalid relocations.

# FIPS and Non-FIPS Symbols

The cryptographic code+data must be included in the hash-verified
data. In general we accomplish that by putting all symbols from
crypto/internal/fips/... packages into the hash-verified data.
But not all.

Note that wrapper code that layers a Go API atop the cryptographic
core is unverified. For example, crypto/internal/fips/sha256 is part of
the FIPS module and verified but the crypto/sha256 package that wraps
it is outside the module and unverified. Also, runtime support like
the implementation of malloc and garbage collection is outside the
FIPS module. Again, only the core cryptographic code and data is in
scope for the verification.

By analogy with these cases, we treat function wrappers like foo·f
(the function pointer form of func foo) and runtime support data like
runtime type descriptors, generic dictionaries, stack maps, and
function argument data as being outside the FIPS module. That's
important because some of them need to be contiguous with other
non-FIPS data, and all of them include data relocations that would be
incompatible with the hash verification.

# Debugging

Bugs in the handling of FIPS symbols can be mysterious. It is very
helpful to narrow the bug down to a specific symbol that causes a
problem when treated as a FIPS symbol. Rather than work that out
manually, if “go test strings” is failing, then you can use

	go install golang.org/x/tools/cmd/bisect@latest
	bisect -compile=fips go test strings

to automatically bisect which symbol triggers the bug.

# Link-Time Hashing

The link-time hash preparation is out of scope for this file;
see ../../link/internal/ld/fips.go for those details.
*/

package obj

import (
	"cmd/internal/objabi"
	"fmt"
	"internal/bisect"
	"internal/buildcfg"
	"log"
	"os"
	"strings"
)

const enableFIPS = true

// IsFIPS reports whether we are compiling one of the crypto/internal/fips/... packages.
func (ctxt *Link) IsFIPS() bool {
	return ctxt.Pkgpath == "crypto/internal/fips" || strings.HasPrefix(ctxt.Pkgpath, "crypto/internal/fips/")
}

// bisectFIPS controls bisect-based debugging of FIPS symbol assignment.
var bisectFIPS *bisect.Matcher

// SetFIPSDebugHash sets the bisect pattern for debugging FIPS changes.
// The compiler calls this with the pattern set by -d=fipshash=pattern,
// so that if FIPS symbol type conversions are causing problems,
// you can use 'bisect -compile fips go test strings' to identify exactly
// which symbol is not being handled correctly.
func SetFIPSDebugHash(pattern string) {
	m, err := bisect.New(pattern)
	if err != nil {
		log.Fatal(err)
	}
	bisectFIPS = m
}

// EnableFIPS reports whether FIPS should be enabled at all
// on the current buildcfg GOOS and GOARCH.
func EnableFIPS() bool {
	// WASM is out of scope; its binaries are too weird.
	// I'm not even sure it can read its own code.
	if buildcfg.GOARCH == "wasm" {
		return false
	}

	// CL 214397 added -buildmode=pie to windows-386
	// and made it the default, but the implementation is
	// not a true position-independent executable.
	// Instead, it writes tons of relocations into the executable
	// and leaves the loader to apply them to update the text
	// segment for the specific address where the code was loaded.
	// It should instead pass -shared to the compiler to get true
	// position-independent code, at which point FIPS verification
	// would work fine. FIPS verification does work fine on -buildmode=exe,
	// but -buildmode=pie is the default, so crypto/internal/fips/check
	// would fail during all.bash if we enabled FIPS here.
	// Perhaps the default should be changed back to -buildmode=exe,
	// after which we could remove this case, but until then,
	// skip FIPS on windows-386.
	//
	// We don't know whether arm or arm64 works, because it is
	// too hard to get builder time to test them. Disable since they
	// are not important right now.
	if buildcfg.GOOS == "windows" {
		switch buildcfg.GOARCH {
		case "386", "arm", "arm64":
			return false
		}
	}

	// AIX doesn't just work, and it's not worth fixing.
	if buildcfg.GOOS == "aix" {
		return false
	}

	return enableFIPS
}

// setFIPSType should be called every time s.Type is set or changed.
// It changes the type to one of the FIPS type (for example, STEXT -> STEXTFIPS) if appropriate.
func (s *LSym) setFIPSType(ctxt *Link) {
	if !EnableFIPS() {
		return
	}

	// Name must begin with crypto/internal/fips, then dot or slash.
	// The quick check for 'c' before the string compare is probably overkill,
	// but this function is called a fair amount, and we don't want to
	// slow down all the non-FIPS compilations.
	const prefix = "crypto/internal/fips"
	name := s.Name
	if len(name) <= len(prefix) || (name[len(prefix)] != '.' && name[len(prefix)] != '/') || name[0] != 'c' || name[:len(prefix)] != prefix {
		return
	}

	// Now we're at least handling a FIPS symbol.
	// It's okay to be slower now, since this code only runs when compiling a few packages.

	// Even in the crypto/internal/fips packages,
	// we exclude various Go runtime metadata,
	// so that it can be allowed to contain data relocations.
	if strings.Contains(name, ".init") ||
		strings.Contains(name, ".dict") ||
		strings.Contains(name, ".typeAssert") ||
		strings.HasSuffix(name, ".arginfo0") ||
		strings.HasSuffix(name, ".arginfo1") ||
		strings.HasSuffix(name, ".argliveinfo") ||
		strings.HasSuffix(name, ".args_stackmap") ||
		strings.HasSuffix(name, ".opendefer") ||
		strings.HasSuffix(name, ".stkobj") ||
		strings.HasSuffix(name, "·f") {
		return
	}

	// This symbol is linknamed to go:fipsinfo,
	// so we shouldn't see it, but skip it just in case.
	if s.Name == "crypto/internal/fips/check.linkinfo" {
		return
	}

	// This is a FIPS symbol! Convert its type to FIPS.

	// Allow hash-based bisect to override our decision.
	if bisectFIPS != nil {
		h := bisect.Hash(s.Name)
		if bisectFIPS.ShouldPrint(h) {
			fmt.Fprintf(os.Stderr, "%v %s (%v)\n", bisect.Marker(h), s.Name, s.Type)
		}
		if !bisectFIPS.ShouldEnable(h) {
			return
		}
	}

	switch s.Type {
	case objabi.STEXT:
		s.Type = objabi.STEXTFIPS
	case objabi.SDATA:
		s.Type = objabi.SDATAFIPS
	case objabi.SRODATA:
		s.Type = objabi.SRODATAFIPS
	case objabi.SNOPTRDATA:
		s.Type = objabi.SNOPTRDATAFIPS
	}
}

// checkFIPSReloc should be called for every relocation applied to s.
// It rejects absolute (non-PC-relative) address relocations when building
// with go build -buildmode=pie (which triggers the compiler's -shared flag),
// because those relocations will be applied before crypto/internal/fips/check
// can hash-verify the FIPS code+data, which will make the verification fail.
func (s *LSym) checkFIPSReloc(ctxt *Link, rel Reloc) {
	if !ctxt.Flag_shared {
		// Writing a non-position-independent binary, so all the
		// relocations will be applied at link time, before we
		// calculate the expected hash. Anything goes.
		return
	}

	// Pseudo-relocations don't show up in code or data and are fine.
	switch rel.Type {
	case objabi.R_INITORDER,
		objabi.R_KEEP,
		objabi.R_USEIFACE,
		objabi.R_USEIFACEMETHOD,
		objabi.R_USENAMEDMETHOD:
		return
	}

	// Otherwise, any relocation we emit must be possible to handle
	// in the linker, meaning it has to be a PC-relative relocation
	// or a non-symbol relocation like a TLS relocation.

	// There are no PC-relative or TLS relocations in data. All data relocations are bad.
	if s.Type != objabi.STEXTFIPS {
		ctxt.Diag("%s: invalid relocation %v in fips data (%v)", s, rel.Type, s.Type)
		return
	}

	// In code, check that only PC-relative relocations are being used.
	// See ../objabi/reloctype.go comments for descriptions.
	switch rel.Type {
	case objabi.R_ADDRARM64, // used with ADRP+ADD, so PC-relative
		objabi.R_ADDRMIPS,  // used by adding to REGSB, so position-independent
		objabi.R_ADDRMIPSU, // used by adding to REGSB, so position-independent
		objabi.R_ADDRMIPSTLS,
		objabi.R_ADDROFF,
		objabi.R_ADDRPOWER_GOT,
		objabi.R_ADDRPOWER_GOT_PCREL34,
		objabi.R_ADDRPOWER_PCREL,
		objabi.R_ADDRPOWER_TOCREL,
		objabi.R_ADDRPOWER_TOCREL_DS,
		objabi.R_ADDRPOWER_PCREL34,
		objabi.R_ARM64_TLS_LE,
		objabi.R_ARM64_TLS_IE,
		objabi.R_ARM64_GOTPCREL,
		objabi.R_ARM64_GOT,
		objabi.R_ARM64_PCREL,
		objabi.R_ARM64_PCREL_LDST8,
		objabi.R_ARM64_PCREL_LDST16,
		objabi.R_ARM64_PCREL_LDST32,
		objabi.R_ARM64_PCREL_LDST64,
		objabi.R_CALL,
		objabi.R_CALLARM,
		objabi.R_CALLARM64,
		objabi.R_CALLIND,
		objabi.R_CALLLOONG64,
		objabi.R_CALLPOWER,
		objabi.R_GOTPCREL,
		objabi.R_LOONG64_ADDR_LO, // used with PC-relative load
		objabi.R_LOONG64_ADDR_HI, // used with PC-relative load
		objabi.R_LOONG64_TLS_LE_HI,
		objabi.R_LOONG64_TLS_LE_LO,
		objabi.R_LOONG64_TLS_IE_HI,
		objabi.R_LOONG64_TLS_IE_LO,
		objabi.R_LOONG64_GOT_HI,
		objabi.R_LOONG64_GOT_LO,
		objabi.R_JMP16LOONG64,
		objabi.R_JMP21LOONG64,
		objabi.R_JMPLOONG64,
		objabi.R_PCREL,
		objabi.R_PCRELDBL,
		objabi.R_POWER_TLS_LE,
		objabi.R_POWER_TLS_IE,
		objabi.R_POWER_TLS,
		objabi.R_POWER_TLS_IE_PCREL34,
		objabi.R_POWER_TLS_LE_TPREL34,
		objabi.R_RISCV_JAL,
		objabi.R_RISCV_PCREL_ITYPE,
		objabi.R_RISCV_PCREL_STYPE,
		objabi.R_RISCV_TLS_IE,
		objabi.R_RISCV_TLS_LE,
		objabi.R_RISCV_GOT_HI20,
		objabi.R_RISCV_PCREL_HI20,
		objabi.R_RISCV_PCREL_LO12_I,
		objabi.R_RISCV_PCREL_LO12_S,
		objabi.R_RISCV_BRANCH,
		objabi.R_RISCV_RVC_BRANCH,
		objabi.R_RISCV_RVC_JUMP,
		objabi.R_TLS_IE,
		objabi.R_TLS_LE,
		objabi.R_WEAKADDROFF:
		// ok
		return

	case objabi.R_ADDRPOWER,
		objabi.R_ADDRPOWER_DS,
		objabi.R_CALLMIPS,
		objabi.R_JMPMIPS:
		// NOT OK!
		//
		// These are all non-PC-relative but listed here to record that we
		// looked at them and decided explicitly that they aren't okay.
		// Don't add them to the list above.
	}
	ctxt.Diag("%s: invalid relocation %v in fips code", s, rel.Type)
}
