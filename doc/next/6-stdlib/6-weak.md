### New weak package

The new [weak](/pkg/weak) package provides weak pointers.

Weak pointers are a low-level primitive provided to enable the
creation of memory-efficient structures, such as weak maps for
associating values, canonicalization maps for anything not
covered by package [unique](/pkg/unique), and various kinds
of caches.
For supporting these use-cases, this release also provides
[runtime.AddCleanup](/pkg/runtime#AddCleanup) and
[maphash.Comparable](/pkg/maphash#Comparable).
