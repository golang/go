# Linker capability ancestry component gate

The exact incoming production file and discovery test from 8afec95580ed24e593307fd9d30d626eff31e22d match the previously reviewed e5 source. Both were supplied through an external overlay to the immutable45 checkout. No tracked source, candidate source, or existing oracle/helper bytes were changed.

The bounded uncached Test(CapGraph|LogosCap) selection passed: 25 top-level tests, 75 reported passing cells, 0 skips, 0 failures. The child discovery oracle exercises 1000 shuffled graphs and the retained root-ordered oracle exercises 3000 graphs. The literal five-second child context remained unchanged.

Component build passed in 11.667s; selected execution in 0.963s; ordinary overlay-aware vet in 1.442s. All commands used one worker, a 512MiB memory limit, disk compilation/cache, and a real home directory for source temporaries.

No mutation was replayed: source matches the reviewed upstream repair, its RED/probe is already recorded, and the existing insertion-by-insertion oracle covers cache invalidation. A final merged toolchain build and installed acceptance remain separate requirements.

The compiled test binary is intentionally excluded from evidence.sha256 and should not be committed. Exact command results, raw logs, overlay inputs, source hashes and retained oracle identities are included.
