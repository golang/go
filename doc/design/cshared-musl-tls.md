# Proposal: Fix c-shared/c-archive dlopen on non-glibc systems

Author: Jay Gowdy
Date: 2026-03-30
Fixes: #13492, #54805
Related: #71953, CL 644975

## Abstract

Go's `-buildmode=c-shared` and `-buildmode=c-archive` produce libraries that
segfault when loaded via `dlopen()` on non-glibc systems (e.g., Alpine
Linux/musl). This proposal fixes two independent issues and adds a user-facing
TLS model control.

## Background

### Issue 1: Garbage argc/argv in .init_array

When a c-shared library is loaded, the ELF dynamic linker runs functions
registered in `.init_array`. Go places `_rt0_<arch>_lib` in `.init_array`,
which saves the register values corresponding to `argc` and `argv`, then
spawns a thread to run `rt0_go`.

Per the ELF gABI specification, `.init_array` functions receive **no
arguments**. glibc non-standardly passes `(argc, argv, envp)` in the
appropriate calling-convention registers. Other libc implementations (musl,
bionic) do not -- the registers contain arbitrary garbage.

The Go runtime subsequently dereferences these garbage pointers in four
locations, causing SIGSEGV:

1. `sysargs()` -- walks argv to find the auxiliary vector
2. `getGodebugEarly()` -- walks argv to find GODEBUG environment variable
3. `goargs()` -- converts argv to Go strings
4. `goenvs_unix()` -- walks argv to find environment variables

### Issue 2: Initial Exec TLS model prevents dlopen

Go's assembler generates Initial Exec (IE) TLS relocations for thread-local
storage when building shared libraries. The IE model requires the dynamic
linker to allocate static TLS space at process startup. Libraries loaded later
via `dlopen()` cannot use IE TLS on musl (and other strict ELF
implementations) because the static TLS allocation has already been finalized.

musl's dynamic linker rejects these libraries with:
```
Error relocating %s: %s: initial-exec TLS resolves to dynamic definition in %s
```

The fix is to use the General Dynamic (GD) TLS model, which calls into the
dynamic linker's TLS resolver at runtime and works correctly with `dlopen()`.

## Design

### Part 1: Runtime glibc detection

A weak reference to `gnu_get_libc_version()` (a glibc-specific function) is
checked during `_cgo_init`, which runs before `sysargs`. This sets a flag
(`_cgo_isglibc`) that the Go runtime reads.

- On glibc: flag is set, argv from `.init_array` is safe to use
- On musl/bionic/other: flag is not set, argv is treated as garbage

When argv is unsafe, the runtime falls back to `/proc/self/auxv` for the
auxiliary vector and `/proc/self/environ` for environment variables. If `/proc`
is unavailable, the runtime uses `mincore`-based page size detection and empty
environment -- both already-supported fallback paths.

The `goargs()` function unconditionally returns an empty `os.Args` in library
mode, since the library does not own the process arguments regardless of libc.

### Part 2: General Dynamic TLS model

A new assembler flag `-tls` controls the TLS model with three values:

- `auto` (default): GD for c-shared/c-archive, IE for other shared modes
- `GD`: Force General Dynamic (TLSDESC on arm64, `__tls_get_addr` on others)
- `IE`: Force Initial Exec (existing behavior)

The `go` command automatically sets `-tls=GD` for c-shared and c-archive
builds on Linux, Android, and FreeBSD. Users targeting glibc-only environments
can override with `-asmflags=-tls=IE` for maximum TLS access performance.

#### Architecture-specific GD TLS sequences

Each architecture has a different instruction sequence for GD TLS. The common
pattern is: compute the address of a GOT entry describing the TLS variable,
call a resolver function, and receive the TLS offset in a designated register.

**arm64** (TLSDESC, 4 instructions, 16 bytes):
```
adrp x0, :tlsdesc:sym       // R_AARCH64_TLSDESC_ADR_PAGE21
ldr  x1, [x0, :lo12:sym]    // R_AARCH64_TLSDESC_LD64_LO12_NC
add  x0, x0, :lo12:sym      // R_AARCH64_TLSDESC_ADD_LO12_NC
blr  x1                      // R_AARCH64_TLSDESC_CALL
// Result: x0 = TLS offset. Clobbers: x0, x1, lr.
```

**amd64** (GD with `__tls_get_addr`, 4 bytes + call):
```
.byte 0x66                    // data16 prefix
leaq sym@tlsgd(%rip), %rdi   // R_X86_64_TLSGD
.word 0x6666                  // data16; data16
rex64
call __tls_get_addr@PLT      // R_X86_64_PLT32
// Result: %rax = address of TLS variable. Clobbers: caller-saved regs.
```

**arm** (GD with `__tls_get_addr`, 3 instructions):
```
ldr  r0, [pc, #offset]       // Load GOT offset for GD entry
bl   __tls_get_addr(PLT)     // R_ARM_TLS_GD32
// Result: r0 = address of TLS variable.
```

**riscv64** (GD with `__tls_get_addr`, 3 instructions):
```
auipc a0, %tls_gd_pcrel_hi(sym)  // R_RISCV_TLS_GD_HI20
addi  a0, a0, %pcrel_lo(label)   // R_RISCV_PCREL_LO12_I
call  __tls_get_addr@plt
// Result: a0 = address of TLS variable.
```

**ppc64/ppc64le** (GD with `__tls_get_addr`, 4 instructions):
```
addis r3, r2, sym@got@tlsgd@ha   // R_PPC64_GOT_TLSGD16_HA
addi  r3, r3, sym@got@tlsgd@l    // R_PPC64_GOT_TLSGD16_LO
bl    __tls_get_addr(sym@tlsgd)   // R_PPC64_TLSGD + R_PPC64_REL24
nop
// Result: r3 = address of TLS variable.
```

**s390x** (GD with `__tls_get_addr`, 3 instructions):
```
lgrl  r2, sym@TLSGD              // R_390_TLS_GD64
brasl r14, __tls_get_addr@PLT    // R_390_TLS_GDCALL
// Result: r2 = address of TLS variable.
```

**loong64** (GD with `__tls_get_addr`, 3 instructions):
```
pcalau12i $a0, %gd_pc_hi20(sym)  // R_LARCH_TLS_GD_PC_HI20
addi.d    $a0, $a0, %got_pc_lo12(sym)  // R_LARCH_GOT_PC_LO12
bl        __tls_get_addr          // R_LARCH_CALL36
// Result: $a0 = address of TLS variable.
```

**mips64** (GD with `__tls_get_addr`):
```
lui   $a0, %tlsgd_hi(sym)        // R_MIPS_TLS_GD (high)
addiu $a0, $a0, %tlsgd_lo(sym)   // R_MIPS_TLS_GD (low)
jalr  $t9, __tls_get_addr
// Result: $v0 = address of TLS variable.
```

#### Runtime assembly changes

The `save_g` and `load_g` functions in each architecture's `tls_<arch>.s` are
modified with `#ifdef TLS_GD` guards. The GD code path must save and restore
any registers clobbered by the GD TLS sequence that callers expect preserved.

For arm64 (TLSDESC), the clobbered registers are X0, X1, and LR. Since
callers of `save_g`/`load_g` may have live values in R1 and LR, these are
saved to scratch registers R10 and R11 across the TLSDESC call.

For architectures using `__tls_get_addr`, the function call follows the
platform's C ABI and clobbers all caller-saved registers. The runtime assembly
must save any live registers accordingly.

### Part 3: Files changed

**Runtime (argv/environ fix + glibc detection):**
- `src/runtime/os_linux.go` -- `sysargs()` guard, `goenvs_lib()`, `libinitArgvSafe()`
- `src/runtime/runtime1.go` -- `goargs()` guard
- `src/runtime/proc.go` -- `getGodebugEarly()` guard
- `src/runtime/cgo/gcc_unix.c` -- weak `gnu_get_libc_version` reference
- `src/runtime/cgo/callbacks.go` -- Go-side `_cgo_isglibc` declaration
- `src/runtime/cgo.go` -- runtime linkname for `_cgo_isglibc`

**Assembler (GD TLS model):**
- `src/cmd/internal/obj/link.go` -- `Flag_tlsgd` field
- `src/cmd/asm/internal/flags/flags.go` -- `-tls` flag definition
- `src/cmd/asm/main.go` -- flag propagation
- `src/cmd/internal/objabi/reloctype.go` -- new `R_<ARCH>_TLS_GD` types
- Per-architecture `cmd/internal/obj/<arch>/`:
  - `a.out.go` -- `C_TLS_GD` addressing class
  - `asm*.go` -- GD instruction encoding + optab entry

**Linker:**
- Per-architecture `cmd/link/internal/<arch>/asm.go` -- GD ELF relocation emission
- `src/cmd/link/internal/ld/data.go` -- GD relocation size handling

**Go command:**
- `src/cmd/go/internal/work/init.go` -- default `-tls=GD` for c-shared/c-archive

**Runtime assembly (per-architecture):**
- `src/runtime/tls_arm64.s`, `tls_arm.s`, `tls_ppc64x.s`, `tls_s390x.s`,
  `tls_riscv64.s`, `tls_loong64.s`, `tls_mipsx.s`, `tls_mips64x.s`
- `src/runtime/asm_amd64.s`, `asm_386.s` (x86 TLS is inline, not in tls_*.s)

**Tests:**
- `src/cmd/cgo/internal/testcshared/cshared_test.go` -- remove Alpine skip
- `src/cmd/cgo/internal/testcarchive/carchive_test.go` -- remove Alpine skip
- `test/cshared-musl/` -- Docker-based end-to-end test harness

## Architecture Status

| Architecture | GD TLS Model | Mechanism | musl dlopen | Status |
|---|---|---|---|---|
| arm64 | Full | TLSDESC | Works | Tested on Alpine |
| amd64 | Full | TLSDESC | Works | Tested (QEMU limited) |
| riscv64 | Full | `__tls_get_addr` GD insns + call reloc | Needs native test | AUIPC+ADDI+CALL sequence |
| arm | Partial | IE insns, LR save only | Not yet | Needs GD instruction encoding |
| ppc64/ppc64le | Partial | IE insns, LR save only | Not yet | Needs GD instruction encoding |
| s390x | Partial | IE insns, LR save only | Not yet | Needs GD instruction encoding |
| loong64 | Partial | IE insns, LR save only | Not yet | Needs GD instruction encoding |
| 386 | Not started | N/A | Not yet | musl/386 is very rare |
| mips/mips64 | N/A | LE only | N/A | No IE/GD support in assembler |

"Partial" means `Flag_tlsgd` routes to IE, runtime assembly saves/restores LR
for future GD support, but the assembler still generates IE instruction
sequences and IE ELF relocations. These architectures will work on glibc but
not musl until proper GD instruction encoding is added. Each needs the
assembler to generate `addi` (compute address) instead of `ld` (load from GOT)
and a `call __tls_get_addr` instead of `add TP`. This is a follow-up CL per
architecture.

## Compatibility

- **glibc systems**: No behavioral change by default. The GD TLS model is
  slightly slower than IE (one indirect call vs. one GOT load per TLS access),
  but the difference is negligible after the first resolution (TLSDESC caches
  the result). Users can force IE with `-asmflags=-tls=IE`.

- **musl systems**: c-shared and c-archive libraries now work correctly with
  `dlopen()`. Previously they segfaulted unconditionally.

- **Non-Linux**: No change. Darwin, Windows, and other platforms are unaffected
  as they use platform-specific TLS mechanisms.

- **PIE executables**: Unaffected. PIE continues to use IE TLS (with IE-to-LE
  optimization for internal linking).

## Testing

1. **Docker-based harness** (`test/cshared-musl/run.sh`): Builds Go from
   source on Alpine (musl) and Ubuntu (glibc), compiles a c-shared library,
   loads it via `dlopen()`, and verifies exported Go functions work correctly.

2. **Existing test suite**: The Alpine skips in `testcshared` and
   `testcarchive` (guarded by `go.dev/issue/19938`) are removed so CI catches
   regressions.

3. **TLS model verification**: `readelf -r` confirms TLSDESC/GD relocations
   in the output .so instead of IE relocations.
