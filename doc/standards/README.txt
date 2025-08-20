Go non-glibc Compatibility Fixes

This directory contains documentation for fixes that enable Go shared
libraries to work correctly on non-glibc Unix systems, particularly for
shared library builds (-buildmode=c-shared and -buildmode=c-archive).

TLS General Dynamic Model (see tls-general-dynamic.txt)
Issue: Go shared libraries fail to load via dlopen() on non-glibc systems
Solution: Comprehensive TLS General Dynamic model implementation across all architectures
Impact: Enables Go shared libraries to work with non-glibc dynamic loaders and libc implementations

argc/argv SIGSEGV Fix (see argc-argv-fix.txt)
Issue: Go shared libraries crash on systems that follow ELF specification strictly
Solution: Added null-safety checks for argc/argv across all Unix platforms
Impact: Prevents SIGSEGV crashes when DT_INIT_ARRAY functions don't receive arguments

Acknowledgments

This work was inspired by and builds upon prior efforts by the Go community:

- Issue #71953: Proposal: runtime: support general dynamic thread local storage model
  (https://github.com/golang/go/issues/71953) - The foundational proposal for TLS General Dynamic support
- Alexander Musman (alexander.musman@gmail.com): ARM64 TLS General Dynamic prototype implementation in
  review 644975 (https://go-review.googlesource.com/c/go/+/644975) that provided the technical foundation
  for this comprehensive multi-architecture implementation
- Issue #73667: Related work that helped identify the scope and approach for comprehensive TLS General Dynamic implementation

Special thanks to the contributors who identified these critical compatibility issues and proposed
solutions that enable Go shared libraries to work correctly across all Unix systems, and to Rich Felker,
author of musl libc, for technical knowledge and documentation on thread local storage models that
informed the TLS General Dynamic implementation approach.

Standards References

ELF Generic Application Binary Interface (gABI)

Link: ELF gABI v4.1 (https://www.sco.com/developers/gabi/latest/contents.html)

Relevant Section 5.2.3 - DT_INIT_ARRAY:
"This element holds the address of an array of pointers to initialization functions..."

Note: The specification does NOT require these functions to receive argc, argv, envp arguments.
Only glibc provides this non-standard extension.

Section 5.1.2 - Dynamic Section:
"The dynamic array tags define the interpretation of the dynamic array entries. The dynamic linker
uses these entries to initialize the process image."

ELF Thread-Local Storage Specification

Link: ELF Handling For Thread-Local Storage (https://www.akkadia.org/drepper/tls.pdf) (Ulrich Drepper)

Section 2.2 - TLS Models:
"General Dynamic: This is the most flexible model. It can be used in all situations, including
shared libraries that are loaded dynamically."

"Initial Exec: This model can be used in shared libraries which are loaded as part of the startup
process of the application."

Section 3.4.1 - x86-64 General Dynamic:
"The general dynamic model is the most general model. It allows accessing thread-local variables
from shared libraries that might be loaded dynamically."

System V Application Binary Interface

x86-64 ABI: System V ABI AMD64 (https://gitlab.com/x86-psABIs/x86-64-ABI)
ARM64 ABI: ARM AAPCS64 (https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst)
RISC-V ABI: RISC-V ELF psABI (https://github.com/riscv-non-isa/riscv-elf-psabi-doc)

Relevance: These specifications define TLS relocations and calling conventions that our TLS General
Dynamic implementation follows.

Additional References

Standards-Compliant libc Implementations:
Most Unix systems use libc implementations that strictly follow specifications rather than providing
glibc-specific extensions. This includes BSD systems, embedded systems, and many containerized environments.

Impact

These fixes enable Go shared libraries to work correctly on:
- Alpine Linux and other lightweight distributions
- FreeBSD, NetBSD, OpenBSD and other BSD variants
- Embedded systems with minimal libc implementations
- Any non-glibc Unix system

The changes maintain full backward compatibility with glibc-based systems while extending support
to non-glibc implementations.