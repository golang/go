### New testing/synctest package

<!-- go.dev/issue/67434, go.dev/issue/73567 -->
The new [testing/synctest](/pkg/testing/synctest) package
provides support for testing concurrent code.

The [synctest.Test] function runs a test function in an isolated
"bubble". Within the bubble, [time](/pkg/time) package functions
operate on a fake clock.

The [synctest.Wait] function waits for all goroutines in the
current bubble to block.
