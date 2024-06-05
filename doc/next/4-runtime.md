## Runtime {#runtime}

The traceback printed by the runtime after an unhandled panic or other
fatal error now indents the second and subsequent lines of the error
message (for example, the argument to panic) by a single tab, so that
it can be unambiguously distinguished from the stack trace of the
first goroutine. See [#64590](/issue/64590) for discussion.
