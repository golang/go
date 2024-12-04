### New benchmark function

Benchmarks may now use the faster and less error-prone [testing.B.Loop] method to perform benchmark iterations like `for b.Loop() { ... }` in place of the typical loop structures involving `b.N` like `for i := n; i < b.N; i++ { ... }` or `for range b.N`. This offers two significant advantages:
 - The benchmark function will execute exactly once per -count, so expensive setup and cleanup steps execute only once.
 - Function call parameters and results are kept alive, preventing the compiler from fully optimizing away the loop body.
