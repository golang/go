package unimported

func _() {
	//@unimported("", bytes, context, cryptoslashrand, time, unsafe, externalpackage)
}

// Create markers for unimported std lib packages. Only for use by this test.
/* bytes */ //@item(bytes, "bytes", "\"bytes\"", "package")
/* context */ //@item(context, "context", "\"context\"", "package")
/* rand */ //@item(cryptoslashrand, "rand", "\"crypto/rand\"", "package")
/* time */ //@item(time, "time", "\"time\"", "package")
/* unsafe */ //@item(unsafe, "unsafe", "\"unsafe\"", "package")
/* pkg */ //@item(externalpackage, "pkg", "\"example.com/extramodule/pkg\"", "package" )
