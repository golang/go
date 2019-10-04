package unimported

func _() {
	//@unimported("", bytes, context, cryptoslashrand, externalpackage, time, unsafe)
}

// Create markers for unimported std lib packages. Only for use by this test.
/* bytes */ //@item(bytes, "bytes", "\"bytes\"", "package")
/* context */ //@item(context, "context", "\"context\"", "package")
/* rand */ //@item(cryptoslashrand, "rand", "\"crypto/rand\"", "package")
/* pkg */ //@item(externalpackage, "pkg", "\"example.com/extramodule/pkg\"", "package" )
/* unsafe */ //@item(unsafe, "unsafe", "\"unsafe\"", "package")
/* time */ //@item(time, "time", "\"time\"", "package")
