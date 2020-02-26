package highlights

import (
	"fmt"         //@mark(fmtImp, "\"fmt\""),highlight(fmtImp, fmtImp, fmt1, fmt2, fmt3, fmt4)
	h2 "net/http" //@mark(hImp, "h2"),highlight(hImp, hImp, hUse)
	"sort"
)

type F struct{ bar int } //@mark(barDeclaration, "bar"),highlight(barDeclaration, barDeclaration, bar1, bar2, bar3)

func _() F {
	return F{
		bar: 123, //@mark(bar1, "bar"),highlight(bar1, barDeclaration, bar1, bar2, bar3)
	}
}

var foo = F{bar: 52} //@mark(fooDeclaration, "foo"),mark(bar2, "bar"),highlight(fooDeclaration, fooDeclaration, fooUse),highlight(bar2, barDeclaration, bar1, bar2, bar3)

func Print() { //@mark(printFunc, "Print"),highlight(printFunc, printFunc, printTest)
	_ = h2.Client{} //@mark(hUse, "h2"),highlight(hUse, hImp, hUse)

	fmt.Println(foo) //@mark(fooUse, "foo"),highlight(fooUse, fooDeclaration, fooUse),mark(fmt1, "fmt"),highlight(fmt1, fmtImp, fmt1, fmt2, fmt3, fmt4)
	fmt.Print("yo")  //@mark(printSep, "Print"),highlight(printSep, printSep, print1, print2),mark(fmt2, "fmt"),highlight(fmt2, fmtImp, fmt1, fmt2, fmt3, fmt4)
}

func (x *F) Inc() { //@mark(xRightDecl, "x"),mark(xLeftDecl, " *"),highlight(xRightDecl, xRightDecl, xUse),highlight(xLeftDecl, xRightDecl, xUse)
	x.bar++ //@mark(xUse, "x"),mark(bar3, "bar"),highlight(xUse, xRightDecl, xUse),highlight(bar3, barDeclaration, bar1, bar2, bar3)
}

func testFunctions() {
	fmt.Print("main start") //@mark(print1, "Print"),highlight(print1, printSep, print1, print2),mark(fmt3, "fmt"),highlight(fmt3, fmtImp, fmt1, fmt2, fmt3, fmt4)
	fmt.Print("ok")         //@mark(print2, "Print"),highlight(print2, printSep, print1, print2),mark(fmt4, "fmt"),highlight(fmt4, fmtImp, fmt1, fmt2, fmt3, fmt4)
	Print()                 //@mark(printTest, "Print"),highlight(printTest, printFunc, printTest)
}

func toProtocolHighlight(rngs []int) []DocumentHighlight { //@mark(doc1, "DocumentHighlight"),mark(docRet1, "[]DocumentHighlight"),highlight(doc1, docRet1, doc1, doc2, doc3, result)
	result := make([]DocumentHighlight, 0, len(rngs)) //@mark(doc2, "DocumentHighlight"),highlight(doc2, doc1, doc2, doc3)
	for _, rng := range rngs {
		result = append(result, DocumentHighlight{ //@mark(doc3, "DocumentHighlight"),highlight(doc3, doc1, doc2, doc3)
			Range: rng,
		})
	}
	return result //@mark(result, "result")
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

func testReturn() bool { //@mark(func1, "func"),mark(bool1, "bool"),highlight(func1, func1, fullRet11, fullRet12),highlight(bool1, bool1, false1, bool2, true1)
	if 1 < 2 {
		return false //@mark(ret11, "return"),mark(fullRet11, "return false"),mark(false1, "false"),highlight(ret11, func1, fullRet11, fullRet12)
	}
	candidates := []int{}
	sort.SliceStable(candidates, func(i, j int) bool { //@mark(func2, "func"),mark(bool2, "bool"),highlight(func2, func2, fullRet2)
		return candidates[i] > candidates[j] //@mark(ret2, "return"),mark(fullRet2, "return candidates[i] > candidates[j]"),highlight(ret2, func2, fullRet2)
	})
	return true //@mark(ret12, "return"),mark(fullRet12, "return true"),mark(true1, "true"),highlight(ret12, func1, fullRet11, fullRet12)
}

func testReturnFields() float64 { //@mark(retVal1, "float64"),highlight(retVal1, retVal1, retVal11, retVal21)
	if 1 < 2 {
		return 20.1 //@mark(retVal11, "20.1"),highlight(retVal11, retVal1, retVal11, retVal21)
	}
	z := 4.3 //@mark(zDecl, "z")
	return z //@mark(retVal21, "z"),highlight(retVal21, retVal1, retVal11, zDecl, retVal21)
}

func testReturnMultipleFields() (float32, string) { //@mark(retVal31, "float32"),mark(retVal32, "string"),highlight(retVal31, retVal31, retVal41, retVal51),highlight(retVal32, retVal32, retVal42, retVal52)
	y := "im a var" //@mark(yDecl, "y"),
	if 1 < 2 {
		return 20.1, y //@mark(retVal41, "20.1"),mark(retVal42, "y"),highlight(retVal41, retVal31, retVal41, retVal51),highlight(retVal42, retVal32, yDecl, retVal42, retVal52)
	}
	return 4.9, "test" //@mark(retVal51, "4.9"),mark(retVal52, "\"test\""),highlight(retVal51, retVal31, retVal41, retVal51),highlight(retVal52, retVal32, retVal42, retVal52)
}

func testReturnFunc() int32 { //@mark(retCall, "int32")
	mulch := 1          //@mark(mulchDec, "mulch"),highlight(mulchDec, mulchDec, mulchRet)
	return int32(mulch) //@mark(mulchRet, "mulch"),mark(retFunc, "int32"),mark(retTotal, "int32(mulch)"),highlight(mulchRet, mulchDec, mulchRet),highlight(retFunc, retCall, retFunc, retTotal)
}
