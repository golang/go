package implementation

import "golang.org/x/tools/internal/lsp/implementation/other"

type ImpP struct{} //@ImpP

func (*ImpP) Laugh() { //@mark(LaughP, "Laugh")
}

type ImpS struct{} //@ImpS

func (ImpS) Laugh() { //@mark(LaughS, "Laugh")
}

type ImpI interface { //@ImpI
	Laugh() //@mark(LaughI, "Laugh"),implementations("Laugh", LaughP, OtherLaughP, LaughS, LaughL, OtherLaughI, OtherLaughS)
}

type Laugher interface { //@implementations("Laugher", ImpP, OtherImpP, ImpI, ImpS, OtherImpI, OtherImpS)
	Laugh() //@mark(LaughL, "Laugh"),implementations("Laugh", LaughP, OtherLaughP, LaughI, LaughS, OtherLaughI, OtherLaughS)
}

type Foo struct {
	other.Foo
}

type U interface {
	U() //TODO: fix flaky @implementations("U", ImpU)
}

type cryer int

func (cryer) Cry(other.CryType) {} //@mark(CryImpl, "Cry")
