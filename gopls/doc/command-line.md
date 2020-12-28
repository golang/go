# Command line

**Note: The `gopls` command-line is still experimental and subject to change at any point.**

`gopls` exposes some (but not all) features on the command-line. This can be useful for debugging `gopls` itself.

<!--TODO(rstambler): Generate this file.-->

Learn about available commands and flags by running `gopls help`.

Much of the functionality of `gopls` is available through a command line interface.

There are two main reasons for this. The first is that we do not want users to rely on separate command line tools when they wish to do some task outside of an editor. The second is that the CLI assists in debugging. It is easier to reproduce behavior via single command.

It is not a goal of `gopls` to be a high performance command line tool. Its command line is intended for single file/package user interaction speeds, not bulk processing.

For more information, see the `gopls` [command line page](command-line.md).
