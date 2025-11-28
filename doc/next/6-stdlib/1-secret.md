### New secret package

<!-- https://go.dev/issue/21865 --->

The new [secret](/pkg/runtime/secret) package is available as an experiment.
It provides a facility for securely erasing temporaries used in
code that manipulates secret information, typically cryptographic in nature.
Users can access it by passing `GOEXPERIMENT=runtimesecret` at build time.

<!-- if we land any code that uses runtimesecret for forward secrecy
like crypto/tls, mention them here too -->

The secret.Do function runs its function argument and then erases all
temporary storage (registers, stack, new heap allocations) used by
that function argument. Heap storage is not erased until that storage
is deemed unreachable by the garbage collector, which might take some
time after secret.Do completes.

This package is intended to make it easier to ensure [forward
secrecy](https://en.wikipedia.org/wiki/Forward_secrecy).
