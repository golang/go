### Directory-limited filesystem access

<!-- go.dev/issue/67002 -->
The new [os.Root] type provides the ability to perform filesystem
operations within a specific directory.

The [os.OpenRoot] function opens a directory and returns an [os.Root].
Methods on [os.Root] operate within the directory and do not permit
paths that refer to locations outside the directory, including
ones that follow symbolic links out of the directory.

- [os.Root.Open] opens a file for reading.
- [os.Root.Create] creates a file.
- [os.Root.OpenFile] is the generalized open call.
- [os.Root.Mkdir] creates a directory.

