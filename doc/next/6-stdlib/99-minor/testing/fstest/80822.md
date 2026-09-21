<!-- go.dev/issue/80822 -->

The `ReadAt` methods of files returned by [testing/fstest.MapFile.Open]
now return [io.EOF] when called with an offset at or beyond the end
of the file, as required by [io.ReaderAt].
