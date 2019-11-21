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

func testForLoops() {
	for i := 0; i < 10; i++ { //@mark(forDecl1, "for"),highlight(forDecl1, forDecl1, brk1, cont1)
		if i > 8 {
			break //@mark(brk1, "break"),highlight(brk1, forDecl1, brk1, cont1)
		}
		if i < 2 {
			for j := 1; j < 10; j++ { //@mark(forDecl2, "for"),highlight(forDecl2, forDecl2, cont2)
				if j < 3 {
					for k := 1; k < 10; k++ { //@mark(forDecl3, "for"),highlight(forDecl3, forDecl3, cont3)
						if k < 3 {
							continue //@mark(cont3, "continue"),highlight(cont3, forDecl3, cont3)
						}
					}
					continue //@mark(cont2, "continue"),highlight(cont2, forDecl2, cont2)
				}
			}
			continue //@mark(cont1, "continue"),highlight(cont1, forDecl1, brk1, cont1)
		}
	}

	arr := []int{}

	for i := range arr { //@mark(forDecl4, "for"),highlight(forDecl4, forDecl4, brk4, cont4)
		if i > 8 {
			break //@mark(brk4, "break"),highlight(brk4, forDecl4, brk4, cont4)
		}
		if i < 4 {
			continue //@mark(cont4, "continue"),highlight(cont4, forDecl4, brk4, cont4)
		}
	}
}
