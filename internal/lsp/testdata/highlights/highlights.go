package highlights

import (
	"fmt"

	"golang.org/x/tools/internal/lsp/protocol"
)

type F struct{ bar int }

var foo = F{bar: 52} //@mark(fooDeclaration, "foo"),highlight(fooDeclaration, fooDeclaration, fooUse)

func Print() { //@mark(printFunc, "Print"),highlight(printFunc, printFunc, printTest)
	fmt.Println(foo) //@mark(fooUse, "foo"),highlight(fooUse, fooDeclaration, fooUse)
	fmt.Print("yo")  //@mark(printSep, "Print"),highlight(printSep, printSep, print1, print2)
}

func (x *F) Inc() { //@mark(xDeclaration, "x"),highlight(xDeclaration, xDeclaration, xUse)
	x.bar++ //@mark(xUse, "x"),highlight(xUse, xDeclaration, xUse)
}

func testFunctions() {
	fmt.Print("main start") //@mark(print1, "Print"),highlight(print1, printSep, print1, print2)
	fmt.Print("ok")         //@mark(print2, "Print"),highlight(print2, printSep, print1, print2)
	Print()                 //@mark(printTest, "Print"),highlight(printTest, printFunc, printTest)
}

func toProtocolHighlight(rngs []protocol.Range) []protocol.DocumentHighlight { //@mark(doc1, "DocumentHighlight"),highlight(doc1, doc1, doc2, doc3)
	result := make([]protocol.DocumentHighlight, 0, len(rngs)) //@mark(doc2, "DocumentHighlight"),highlight(doc2, doc1, doc2, doc3)
	kind := protocol.Text
	for _, rng := range rngs {
		result = append(result, protocol.DocumentHighlight{ //@mark(doc3, "DocumentHighlight"),highlight(doc3, doc1, doc2, doc3)
			Kind:  kind,
			Range: rng,
		})
	}
	return result
}
