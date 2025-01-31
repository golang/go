## Runtime {#runtime}

<!-- go.dev/issue/71517 -->

The message printed when a program exits due to an unhandled panic
that was recovered and re-raised no longer repeats the text of
the panic value.

Previously, a program which panicked with `panic("PANIC")`,
recovered the panic, and then re-panicked with the original
value would print:

    panic: PANIC [recovered]
      panic: PANIC

This program will now print:

    panic: PANIC [recovered, reraised]
