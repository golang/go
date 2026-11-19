## Runtime {#runtime}

### Memory profile {#memory-profile}
<!-- go.dev/issue/79179 -->

A new profile type that breaks down total process memory (RSS) is now available.
The new profile type, named `memory`, is supported in the [runtime/pprof] package
and as a [net/http/pprof] endpoint, `/debug/pprof/memory`.

Go developers are often confused when the operating system reports much more
memory usage than the heap profile shows.
That's because the existing `inuse_space` heap profile only shows a sample of
objects that were live at the most recently completed garbage collection.
With the default garbage collection configuration, this is often less than
half of RSS.

The new memory profile is a superset of the `inuse_space` heap profile that
closes this gap by breaking down RSS into Go and non-Go memory.
Go memory is further split into the heap, stacks, and runtime metadata.
The heap includes the same live object samples as the heap profile, as well
as dead objects the most recently completed garbage collection found
unreachable.

Special thanks to Felix Geisendörfer and Nick Ripley at Datadog for
contributing this work.
