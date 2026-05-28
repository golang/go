## Compiler {#compiler}

The compiler now resolves a relative filename in a `//line` or `/*line*/`
directive against the directory of the file containing the directive,
matching [go/scanner]. Absolute filenames are unaffected.
See [#70478](/issue/70478).

## Assembler {#assembler}

## Linker {#linker}


